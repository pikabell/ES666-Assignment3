import pdb
import src
import glob
import importlib.util
import os
import cv2



### Change path to images here
path = 'Images{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters
###

all_submissions = glob.glob('./src/*')
os.makedirs('./results/', exist_ok=True)
for idx,algo in enumerate(all_submissions):
    print('****************\tRunning Awesome Stitcher developed by: {}  | {} of {}\t********************'.format(algo.split(os.sep)[-1],idx+1,len(all_submissions)))
    try:
        module_name = '{}_{}'.format(algo.split(os.sep)[-1],'stitcher')
        filepath = '{}{}stitcher.py'.format( algo,os.sep,'stitcher.py')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PanaromaStitcher = getattr(module, 'PanaromaStitcher')
        inst = PanaromaStitcher()

        ###
        for impaths in glob.glob(path):
            print('\t\t Processing... {}'.format(impaths))
            stitched_image, homography_matrix_list , int_list = inst.make_panaroma_for_images_in(path=impaths)

            outfile =  './results/{}/{}.png'.format(impaths.split(os.sep)[-1],spec.name)
            os.makedirs(os.path.dirname(outfile),exist_ok=True)
            cv2.imwrite(outfile,stitched_image)
            for i in range(len(int_list)):
                cv2.imwrite('./results/{}/stitch_{}.png'.format(impaths.split(os.sep)[-1],i),int_list[i])
            print(homography_matrix_list)
            # create a file to write homography matrix
            with open('./results/{}/homography_matrix.txt'.format(impaths.split(os.sep)[-1]), 'w') as f:
                for item in homography_matrix_list:
                    f.write("%s\n" % item)
            print('Panaroma saved ... @ {}'.format(outfile))
            print('\n\n')

    except Exception as e:
        print('Oh No! My implementation encountered this issue\n\t{}'.format(e))
        print('\n\n')
