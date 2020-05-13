'''
Created on Mar, 2018

@author: mgulcin
'''

import csv
from gensim.models.fasttext import FastText as FT_gensim
import os
import time

import Common


def createFastTextModel(data, isSG, vectorSize, maxNgram, modelFilePath):
    # train model
    model_gensim = FT_gensim(sg=isSG, size=vectorSize, min_count=1, min_n=1, max_n=maxNgram, compatible_hash=False)
    # build the vocabulary
    model_gensim.build_vocab(data)
    # train the model
    model_gensim.train(data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)    
    #save
    model_gensim.save(modelFilePath)

    
def trainAndSaveModel(modelFolder, baseModelFileName, timeInfoFolder, timeInfoFile, 
                      data, isSG, vectorSize, maxNgram):
        start_time = time.time()
        usedFeaturesAcronym = Common.createUsedFeaturesAcronomy(isSG=isSG, vectorSize=vectorSize, maxNgram=maxNgram)
        modelFilePath = os.path.join(modelFolder, baseModelFileName + usedFeaturesAcronym)
        createFastTextModel(data=data, isSG=isSG, vectorSize=vectorSize, maxNgram=maxNgram,
                       modelFilePath=modelFilePath) 
        Common.printTime((time.time() - start_time), usedFeaturesAcronym, timeInfoFolder, timeInfoFile)
        
def run(trainDataFilePath, modelFolder, baseModelFileName, timeInfoFolder, timeInfoFile):
    # read the training data
    user_2_locations = Common.readTrainData(trainDataFilePath)
    data = user_2_locations.values()
    
    # train and save the model
    vectorSizes = [10, 50, 100, 150, 200, 250]
    isSGValues = [0, 1]
    maxngram_values = [5, 9]#range(0, 11, 1)
    for isSG in isSGValues:
        # for vectorSize in vectorSizes:
        #     trainAndSaveModel(modelFolder, baseModelFileName, timeInfoFolder, timeInfoFile,
        #                       data=data, isSG=isSG, vectorSize=vectorSize, maxNgram=5)
                         
        for maxNgram in maxngram_values:
            # maxNgram == 5 is already created on the above loop(s)
            # if maxNgram == 5:
            #     continue
            trainAndSaveModel(modelFolder, baseModelFileName, timeInfoFolder, timeInfoFile,
                              data=data, isSG=isSG, vectorSize=100, maxNgram=maxNgram)

if __name__ == '__main__':
    base_dir = "XXX"
    dataFolder = "XXX/dataset/checkins2011_janjuly/"
    trainDataFilePath = os.path.join(dataFolder, "user_2_list_of_visits_2011_05_01_train.csv")
    modelFolder = base_dir + "model/model_janjuly/model_fasttext/"
    baseModelFileName = "fasttext_locs_only_model"
    timeInfoFolder = base_dir + "/model/model_janjuly/model_fasttext/"
    timeInfoFile = "time_fasttext_locs_only_model.txt"
    run(trainDataFilePath, modelFolder, baseModelFileName, timeInfoFolder, timeInfoFile)

    

