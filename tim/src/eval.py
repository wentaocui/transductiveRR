import numpy as np
from utils import warp_tqdm, compute_confidence_interval, load_checkpoint
from utils import load_pickle, save_pickle
from utils import write_log
from datasets.ingredient import get_dataloader
import os
import torch
import torch.nn.functional as F
from tim import TIM, TIM_ADM, TIM_GD
from transduction import transductive_encode, iterative_transductive_encode

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

class Evaluator:
    def __init__(self, device, params):
        self.device = device
        self.params = params

    def run_full_evaluation(self, model, model_path, model_tag, shot, method, callback, params):
        """
        Run the evaluation over all the tasks in parallel
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            method : Which method to use for inference ("baseline", "tim-gd" or "tim-adm")
            shot : Number of support shot to try

        returns :
            results : List of the mean accuracy for each number of support shot
        """
        print("=> Runnning full evaluation with method: {}".format(method))

        # Load pre-trained model
        load_checkpoint(model=model, model_path=model_path, type=model_tag)

        # Get loaders
        loaders_dic = self.get_loaders(params.eval_used_set, params.eval_target_data_path, params.eval_target_split_dir, params)

        # Extract features (just load them if already in memory)
        extracted_features_dic = self.extract_features(model, model_path, model_tag, params.eval_used_set, loaders_dic)
        results = []

        tasks = self.generate_tasks(extracted_features_dic, shot, params.eval_number_tasks, params)
        logs = self.run_task(tasks, model, callback, params)

        l2n_mean, l2n_conf = compute_confidence_interval(logs['acc'][:, -1])

        print('============> Meta Test: {} \nfeature\tL2N\n    {}-shot     \t{:.4f}({:.4f})'.format(
              model_tag.upper(), shot, l2n_mean, l2n_conf))
        results.append(l2n_mean)
        write_log(params, shot, l2n_mean, l2n_conf)
        return results

    def run_task(self, task_dic, model, callback, params):

        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model, params.eval_method)

        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        z_s, z_q = task_dic['z_s'], task_dic['z_q']

        # Represent each query and support instance transductively
        if params.enable_transduct == True:
            if params.iterative_transduct == True:
                z_s, z_q = iterative_transductive_encode(z_s, z_q, params)
            else:
                z_s, z_q = transductive_encode(z_s, z_q, params)

        # Perform normalizations required
        z_s = F.normalize(z_s, dim=2)
        z_q = F.normalize(z_q, dim=2)

        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s)
        tim_builder.init_weights(support=support, y_s=y_s, query=query, y_q=y_q)

        # Run adaptation
        tim_builder.run_adaptation(support=support, query=query, y_s=y_s, y_q=y_q, callback=callback)

        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return logs

    def get_tim_builder(self, model, method):
        # Initialize TIM classifier builder
        tim_info = {'temp': self.params.tim_temp, 'loss_weights': self.params.tim_loss_weights,
                    'iter': self.params.tim_iter, 'lr': self.params.tim_lr, 'model': model}
        if method == 'tim_adm':
            tim_builder = TIM_ADM(**tim_info)
        elif method == 'tim_gd':
            tim_builder = TIM_GD(**tim_info)
        elif method == 'baseline':
            tim_builder = TIM(**tim_info)
        else:
            raise ValueError("Method must be in ['tim_gd', 'tim_adm', 'baseline']")
        return tim_builder

    def get_loaders(self, used_set, target_data_path, target_split_dir, params):
        # First, get loaders
        loaders_dic = {}
        loader_info = {'aug': False, 'shuffle': False, 'out_name': False}

        if target_data_path is not None:  # This mean we are in the cross-domain scenario
            path = target_data_path
            split_dir = target_split_dir
            # loader_info.update({'path': target_data_path,
            #                     'split_dir': target_split_dir})
        else:
            path = params.dataset_path
            split_dir = params.dataset_split_dir

        train_loader = get_dataloader('train', params.dataset_enlarge,
                                      params.dataset_num_workers, params.dataset_batch_size,
                                      params.dataset_disable_random_resize, path,
                                      split_dir, params.dataset_jitter,
                                      **loader_info)
        loaders_dic['train_loader'] = train_loader

        test_loader = get_dataloader(used_set, params.dataset_enlarge,
                                      params.dataset_num_workers, params.dataset_batch_size,
                                      params.dataset_disable_random_resize, params.dataset_path,
                                      params.dataset_split_dir, params.dataset_jitter,
                                     **loader_info)
        loaders_dic.update({'test': test_loader})
        return loaders_dic

    def extract_features(self, model, model_path, model_tag, used_set, loaders_dic):
        """
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            used_set : Set used between 'test' and 'val'
            n_ways : Number of ways for the task

        returns :
            extracted_features_dic : Dictionnary containing all extracted features and labels
        """

        # Load features from memory if previously saved ...
        save_dir = os.path.join(model_path, model_tag, used_set)
        filepath = os.path.join(save_dir, 'output.plk')
        if os.path.isfile(filepath):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic

        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        model.eval()
        with torch.no_grad():

            all_features = []
            all_labels = []
            for i, (inputs, labels, _) in enumerate(warp_tqdm(loaders_dic['test'], False)):
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs, True)
                all_features.append(outputs.cpu())
                all_labels.append(labels)
            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            extracted_features_dic = {'concat_features': all_features,
                                      'concat_labels': all_labels
                                      }
        print(" ==> Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic

    def get_task(self, shot, n_ways, query_shots, extracted_features_dic):
        """
        inputs:
            extracted_features_dic : Dictionnary containing all extracted features and labels
            shot : Number of support shot per class
            n_ways : Number of ways for the task

        returns :
            task : Dictionnary : z_support : torch.tensor of shape [n_ways * shot, feature_dim]
                                 z_query : torch.tensor of shape [n_ways * query_shot, feature_dim]
                                 y_support : torch.tensor of shape [n_ways * shot]
                                 y_query : torch.tensor of shape [n_ways * query_shot]
        """
        all_features = extracted_features_dic['concat_features']
        all_labels = extracted_features_dic['concat_labels']
        all_classes = torch.unique(all_labels)
        samples_classes = np.random.choice(a=all_classes, size=n_ways, replace=False)
        support_samples = []
        query_samples = []
        for each_class in samples_classes:
            class_indexes = torch.where(all_labels == each_class)[0]
            indexes = np.random.choice(a=class_indexes, size=shot + query_shots, replace=False)
            support_samples.append(all_features[indexes[:shot]])
            query_samples.append(all_features[indexes[shot:]])

        y_support = torch.arange(n_ways)[:, None].repeat(1, shot).reshape(-1)
        y_query = torch.arange(n_ways)[:, None].repeat(1, query_shots).reshape(-1)

        z_support = torch.cat(support_samples, 0)
        z_query = torch.cat(query_samples, 0)

        task = {'z_s': z_support, 'y_s': y_support,
                'z_q': z_query, 'y_q': y_query}
        return task

    def generate_tasks(self, extracted_features_dic, shot, number_tasks, params):
        """
        inputs:
            extracted_features_dic :
            shot : Number of support shot per class
            number_tasks : Number of tasks to generate

        returns :
            merged_task : { z_support : torch.tensor of shape [number_tasks, n_ways * shot, feature_dim]
                            z_query : torch.tensor of shape [number_tasks, n_ways * query_shot, feature_dim]
                            y_support : torch.tensor of shape [number_tasks, n_ways * shot]
                            y_query : torch.tensor of shape [number_tasks, n_ways * query_shot] }
        """
        print(f" ==> Generating {number_tasks} tasks ...")
        tasks_dics = []
        for _ in warp_tqdm(range(number_tasks), False):
            task_dic = self.get_task(shot, params.eval_n_ways, params.eval_query_shots, extracted_features_dic)
            tasks_dics.append(task_dic)

        # Now merging all tasks into 1 single dictionnary
        merged_tasks = {}
        n_tasks = len(tasks_dics)
        for key in tasks_dics[0].keys():
            n_samples = tasks_dics[0][key].size(0)
            merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks, n_samples, -1)
        return merged_tasks
