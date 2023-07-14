from PIL import Image 
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error
from common.clickme_dataset import get_stimuli_paths,get_human_data
from common.utils import im_crop_center, get_synset

def run_images(model,preprocess,model_name=''):
    """
    Function to run the images using the Psychophysics stimuli. 

    Parameters : 

      * Model : model to be evaluated. 
      * preprocess: Preprocessing function to be used to preprocess data
      
    """
    results = []
    stimuli = get_stimuli_paths()
    synmap,labels,revmap = get_synset()
    
    for f in stimuli:
        file = f.split('/')[-1]
        label = ''.join(file[:-5].split('_')[1:])
        indx_label = revmap[label]['index']
        # Anything below 398 is animal, else non-animal. 
        if indx_label <398: 
            task_label = 1
        else: 
            task_label = 0
        diff = file.split('_')[0]
        sample = file[-5]
        img = Image.open(f)
        img = im_crop_center(img,224,224)
        img = np.array(img)
        img2 = np.stack((img,)*3, axis=-1)
       
        img = np.stack((img,)*3, axis=-1)
        img = preprocess(img)
        
        output = model.predict(np.array([img]))
        imagenet_indx = np.argmax(output[0])
  
        if imagenet_indx <398: 
            task = 1
        else: 
            task = 0 
        logits_score = np.max(output[0])
        animalness = np.sum(output[0][:398])  
        objecteness = np.sum(output[0][398:])
        decisioness = animalness - objecteness 
        animalness_mean = np.mean(output[0][:398])  
        objecteness_mean = np.mean(output[0][398:])
        decisioness_mean = animalness_mean - objecteness_mean 
        #print(decisioness>0)
        results.append([f,file,label,diff,sample,indx_label,task_label,model_name,imagenet_indx,task,logits_score,labels[imagenet_indx],animalness,objecteness,int(decisioness>0),int(decisioness_mean>0)])
    results = pd.DataFrame(results, columns=['path','name','label','difficulty','sample number','imagenet_index_label','task_label','model','output','task output','conf','output_imagenet_label','animalness','objecteness','decisionness','decisioness_mean'])
    
    return results 

def parsing_results(results):
    """
    Function to parse the results of the Psychophysics stimuli.
    
    Parameters :
        * results : Results of the Psychophysics stimuli.

    Returns :

        * parsing : Parsed results of the Psychophysics stimuli.
        

    """

    xx = [1.0,
          1.5848931924611136,
          2.51188643150958,
          3.981071705534973,
          6.309573444801933,
          10.0,
          15.848931924611142,
          25.11886431509581,
          39.810717055349734,
          63.09573444801933,
          100.0]
  
    difficulty = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90','full']
    dictionary= {str(k):v for k,v in zip(difficulty,xx)}
    parsing= {}
    m = results.model.unique()[0]
    parsing[m]= {
        'x':[],
        'y':[],
        'z':[]
    }
    for d in difficulty:
      t = results[results['difficulty']==d]
      parsing[m]['x'].append(dictionary[d])
      parsing[m]['y'].append(accuracy_score(t['task_label'].tolist(),t['task output'].tolist()))
      parsing[m]['z'].append(accuracy_score(t['task_label'].tolist(),t['decisioness_mean'].tolist()))

    return parsing


def psychophysics_score(parsing,gt,iters=1000):
    """
    Function to compute the Psychophysics score.
    

    Parameters :
        * parsing : Parsed results of the Psychophysics stimuli.    
        * gt : Ground truth of the Psychophysics stimuli.
        * iters : Number of iterations to compute the Psychophysics score.

    Returns :

        * conf_bars : Psychophysics score.
    """
    conf_bars ={}
    for m in parsing:
      x = np.array(parsing[m]['x'])
      y =  (parsing[m]['y']-np.min(parsing[m]['y']))/(np.max(parsing[m]['y'])-np.min(parsing[m]['y']))
      scores =[]
      
      for i in range(iters):
          resample_idx  = np.random.randint(0, len(gt), size=len(gt))
          sample_X = x[resample_idx]
          sample_y = gt[resample_idx]
          scores.append(-np.log(mean_squared_error(sample_X, sample_y)))
    conf_bars[m]={'mean':np.mean(scores),'std':np.std(scores)}
    return conf_bars 

def run_psychophysics_benchmark(model,preprocess):
    """
    Function to run the Psychophysics benchmark.

    Parameters : 

      * Model : model to be evaluated. 
      * preprocess: Preprocessing function to be used to preprocess data
      
    """

    result_images = run_images(model,preprocess)
    parsing = parsing_results(result_images)
    mpx, mpy, mpz = get_human_data()
    conf_bars = psychophysics_score(parsing,mpy)
    print('Psychophysics score: ',conf_bars)
    return conf_bars

