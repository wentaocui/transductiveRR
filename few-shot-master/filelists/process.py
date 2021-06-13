import sys
import os
from subprocess import call

if len(sys.argv) != 2:
  raise Exception('Incorrect command! e.g., python3 process.py DATASET [cars, cub, places, miniImagenet, plantae]')
dataset = sys.argv[1]

print('--- process ' + dataset + ' dataset ---')
if not os.path.exists(os.path.join(dataset, 'source')):
  os.makedirs(os.path.join(dataset, 'source'))
os.chdir(os.path.join(dataset, 'source'))

# download files
if dataset == 'miniImagenet':
  # this file is from MAML++: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
  call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/filelists/mini_imagenet_full_size.tar.bz2', shell=True)
  call('tar -xjf mini_imagenet_full_size.tar.bz2', shell=True)
else:
  raise Exception('No such dataset!')

# process file
os.chdir('..')
call('python3 write_' + dataset + '_filelist.py', shell=True)