# Machine Learning for Predicting and Maximizing the Response of Breast Cancer Patients to Neoadjuvant Therapy

The results in the paper can be replicated with the following steps: 

1. Download all files in this repository and save them in a directory on your computer;  download the I-SPY2 gene expression data from GEO using the accession number GSE194040; you will get a file with the following name, save it in the same directory: GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt
2. Run the notebook code_explore_data.ipynb for data processing.  It will output a data file named data_gexp4000.csv that contains expression values of 4,000 genes with the largest variance. 
3. Run the script code_xgb_v35.py for training and testing XGBoost models.  The results we obtained are in the file folder res_xgb35.
4. Run the notebook code_plot_roc.ipynb to plot ROC curves (Figure 1 in the paper).
5. Run the notebook code_plot_precall.ipynb to plot RR curves (Supplemental Figure S1 in the paper) and PPT curves (Figure 3 in the paper).
6. Run the notebook code_plot_acc_allF.ipynb to compute performance metrics AUC, accuracy, sensitivity, and specificity in Table 2 of the paper, and pCR rate results in Table 3 of the paper.
7. Run the notebook code_analysis_max.ipynb to plot histograms in Figure 2 of the paper, and compute the pCR results in Table 4 of the paper. 
8. Run the notebook code_gexpa.ipynb to plot the gene expression values in Figure 4 of the paper. 
   
