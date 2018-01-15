
# coding: utf-8

# In[ ]:

from os import path
import sys
python_location = path.dirname(sys.executable)+'/python'
basic_program = open('Model_Time_Series.py', 'r').read()


# In[ ]:

from os import system

target_predictors = [
                    'Citations_Backward_N', 
                    'Citations_Backward_Age_Mean', 'Citations_Backward_Age_STD', 
                    'meanSPNPcited_1year_before', 'stdSPNPcited_1year_before',
                    'N_Patents',
                    ]

data_types = ['Performance']#, 'Price']
model_type = 'VAR_separate'

for data_type in data_types:
    for target_predictor in target_predictors:
        header = """#!{3}
#PBS -l nodes=1:ppn=4
#PBS -l walltime=20:00:00
#PBS -l mem=10000m
#PBS -N {0}_{1}_{2}

data_type = '{0}'
target_predictor = '{1}'
model_type = '{2}'
    """.format(data_type, target_predictor, model_type, python_location)

        options = """""".format()

        this_program = header+options+basic_program

        this_job_file = 'jobfiles/run_{0}_{1}_{2}.py'.format(model_type, data_type, target_predictor)


        f = open(this_job_file, 'w')
        f.write(this_program)
        f.close()

        system('qsub '+this_job_file)

