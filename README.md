# Machine Learning for Predicting and Maximizing the Response of Breast Cancer Patients to Neoadjuvant Therapy

The results in the paper can be replicated with the following steps: 

1. Download all files in this repository and save them in a directory on your computer;  download the I-SPY2 gene expression data from GEO using the accession number GSE194040; you will get a file with the following name, save it in the same directory: GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt
2. Run the notebook code_explore_data.ipynb for data processing.  It will output a data file named data_gexp4000.csv that contains expression values of 4,000 genes with the largest variance. 
3. Run the script code_xgb_v35.py for training and testing XGBoost models.  
4. aaa
   
