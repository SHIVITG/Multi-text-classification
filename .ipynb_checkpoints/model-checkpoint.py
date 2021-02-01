from  utlis.author_prediction  import  AuthPredict

#arguments to be parsed from command line 
import  argparse 
ap  =  argparse . ArgumentParser () 

ap . add_argument ( "--file" ,  required = False ,  type = str ,  help = "train file for training/test file for prediction" ) 
ap . add_argument ( "--opt" ,  required = False ,  type = str, help = "values that can be passed: training/prediction")

args  =  ap . parse_args () 

# file location
if  args . file : 
    file = args.file

# option of training/prediction
if  args . opt : 
    opt = args.opt
    
author_predict = AuthPredict()
def model_exec(opt,file):
    if opt == "training":
        author_predict.training(file)
    elif opt == "prediction":
        author_predict.prediction(file)
    else:
        print("Invalid process entered via command line")



if  __name__  ==  "__main__" : 

    #call function
    model_exec( args . opt,args . file )