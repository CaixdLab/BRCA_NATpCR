# Machine Learning for Predicting and Maximizing the Response of Breast Cancer Patients to Neoadjuvant Therapy

The results in the paper can be replicated by following these steps: 

1. Download all files from this repository and save them in a designated directory on your computer. Next, download the I-SPY2 gene expression data from the GEO database using the accession number GSE194040. You will receive a file named **GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt**; please save it in the same directory.
2. Run the notebook **code_explore_data.ipynb** for data processing. This will generate a data file named **data_gexp4000.csv**, which contains the expression values of the 4,000 genes with the highest variance. 
3. Execute the script **code_xgb_v35.py** to train and test the XGBoost models. The results we have obtained with the script can be found in the **res_xgb35** folder within this repository.
4. Run the notebook **code_plot_roc.ipynb** to generate the ROC curves illustrated in Figure 1 of the paper.  
5. Execute the notebook **code_plot_precall.ipynb** to plot the PR curves shown in Supplemental Figure S1 and the PPT curves depicted in Figure 3 of the paper.
6. Execute the notebook **code_plot_acc_allF.ipynb** to calculate the performance metrics—AUC, accuracy, sensitivity, specificity, and F1-score—displayed in Table 2 of the paper, as well as the pCR rate results shown in Table 3.
7. Run the notebook **code_analysis_max.ipynb** to generate histograms illustrated in Figure 2 of the paper and to compute the pCR results presented in Table 4.
8. Execute the notebook **code_analysis_frankq.ipynb** to identify the top 100 genes associated with pCR for each of the ten I-SPY2 arms, which are included in Supplemental File 2.
9.  Run the notebook **code_gexpa.ipynb** to visualize the gene expression values depicted in Figure 4 of the paper.

   
