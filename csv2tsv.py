import pandas as pd

df = pd.read_csv("output/submissiontrain_180131_1_ts0536_seed0_val_loss0.48878595829_val_acc0.836666667461.csv",
                 index_col=False, header=None)
print(df.head())
df.to_csv("output/submissiontrain_180131_1_ts0536_seed0_val_loss0.48878595829_val_acc0.836666667461.tsv", sep='\t', index=False, header=False)