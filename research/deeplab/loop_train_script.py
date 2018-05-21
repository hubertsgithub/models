from argparse import ArgumentParser
import shutil
import os
import subprocess

parser = ArgumentParser()
parser.add_argument('script', default=None, help='training script to run')
parser.add_argument('--num_iterations', type=int, default=100000, help='number of iterations to run')
parser.add_argument('--start_iteration', type=int, default=0, help='iteration to start on')
args = parser.parse_args()

args.script = os.path.abspath(args.script)

num_it = args.num_iterations
start_iteration = args.start_iteration
eval_it = 10000 # Evaluate every 10000
for i in range(start_iteration+eval_it, num_it+1, eval_it):
    print '#######################################'
    print 'Setting NUM_ITERATIONS TO: {}'.format(i)
    print '#######################################'
    print 'Calling script {} {}...'.format(args.script, i)
    subprocess.call(['sh', args.script, str(i)])

