import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Load both datasets
customer = pd.read_csv("Customers_rfm.csv")
order = pd.read_csv("Orders_rfm.csv")

#Inspect
print(f"FOR customers\n\n Shape is {customer.shape}\n")
print(f"Column is {customer.columns}\n")
print(f"Data-type is {customer.dtypes}\n")
print(f"Duplicates is {customer.duplicated().sum()}\n")
print(f"NULL Before Cleaning\n {customer.isnull().sum()}\n\n")

print(f"FOR orders\n\n Shape is {order.shape}\n")
print(f"Column is {order.columns}\n")
print(f"Data-type is {order.dtypes}\n")
print(f"Duplicates is {order.duplicated().sum()}\n")
print(f"NULL Before Cleaning\n {order.isnull().sum()}\n")

#Merge both Dataset
df = pd.merge(order, customer, on="customer_id", how="left", indicator=True)
print(f"After merging customers and orders shape of our dataset is {df.shape}\n")
print(df)
print("\n")

#Validate joins
print("customers match\n", df["_merge"].value_counts())
print("\n")
print("orders match\n", df[df["_merge"] != "both"])
print("\n")

#Drop Duplicates
print("Before:", df.shape)
df = df.drop_duplicates()     
print("After:",df.shape)   
print("\n")

# Null handeling
print("NULL Before Cleaning:\n")
print(df.isnull().sum())
print("\n")

#df = df.dropna(subset=["category", "email", "age"])

df["category"] = df["category"].fillna("unknown")
df["email"] = df["email"].fillna("Not Known")
df["age"] = df["age"].fillna(df["age"].median())
print("NULL After Cleaning:\n", df.isnull().sum())
print("\n") 

#Trim Space
# Check columns with extra spaces
for col in df.select_dtypes(include="object").columns:
    dirty_values = df[df[col] != df[col].str.strip()][col].unique()
    if len(dirty_values) > 0:
        print(f"Extra spaces found in column: {col}\n")
        print(dirty_values)
# Remove extra spaces
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
print("Spaces cleaned successfully\n")

#Standardize casing
title = ["city", "acquisition_channel", "category", "order_status"]
for i in title:
    df[i] = df[i].str.title()
print("Standardize casing Succesfully\n")


#Fix invalid categories
# Check unique categories
print(df["category"].unique())

# Valid categories
valid_categories = ["Electronics", "Fashion", "Home"]

# Replace invalid categories with Unknown
df.loc[~df["category"].isin(valid_categories), "category"] = "Unknown"

# Verify
print(df["category"].value_counts())
print("\n")

#Identify Invalid Quantities  
invalid_qty = df[df["quantity"] <= 0]
print(f"Invalid Quantity is\n {invalid_qty}\n")

#RFM MATRICES
#For only complete orders
completed_orders = df[df["order_status"] == "Completed"]
completed_orders["order_date"] = pd.to_datetime(completed_orders["order_date"])

#Today Date 
today = pd.Timestamp.today()

#Recency 
Recency = completed_orders.groupby("customer_id")["order_date"].max()
Recency = (today - Recency).dt.days
Recency = Recency.reset_index()
Recency.columns = ["customer_id", "Recency"]

#Frequency
Frequency = completed_orders.groupby("customer_id")["order_id"].count()
Frequency = Frequency.reset_index()
Frequency.columns = ["customer_id", "Frequency"]

#Total revenue generated of complete order [re𝑣𝑒𝑛𝑢𝑒 = 𝑄𝑢𝑎𝑛𝑡𝑖𝑡𝑦 ×𝑈𝑛𝑖𝑡𝑃𝑟𝑖𝑐𝑒 ×(1−𝐷𝑖𝑠𝑐𝑜𝑢𝑛𝑡)] 
completed_orders["revenue"] = completed_orders["quantity"] * completed_orders["unit_price"] * (1-completed_orders["discount_pct"])
revenue = completed_orders["revenue"].sum()
print("Total Revenue of complete order is:\n",revenue)

#Monetary
Monetary = completed_orders.groupby("customer_id")["revenue"].sum()
Monetary = Monetary.reset_index()
Monetary.columns = ["customer_id", "Monetary"]

#Merge both
rfm = pd.merge(Recency, Frequency, on="customer_id")
print("After Merge:\n",rfm.head(10))

rfm = pd.merge(rfm, Monetary, on="customer_id")
print("After Merge:\n",rfm.head(10))

# Recency Score
rfm["R_score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5,4,3,2,1]
#          good --> bad
)
# Frequency Score
rfm["F_score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1,2,3,4,5]
#          bad --> good
)
# Monetary Score
rfm["M_score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1,2,3,4,5]
#          bad --> good
)
print("RFM Score\n",rfm)

#Customer Segmentation
#Function
def segment_customer(row):
    if row["R_score"] >= 4 and row["F_score"] >= 4 and row["M_score"] >= 4:
        return "Champions"
    
    elif row["R_score"] >= 3 and row["F_score"] >= 3 and row["M_score"] >= 3:
        return "Loyal Customer"
    
    if row["R_score"] >= 4 and row["F_score"] <= 3:
        return "Potential Loyalists"
    
    if row["R_score"] <= 2 and ( row["F_score"] >= 3 or row["M_score"] >= 3):
        return "At Risk"

    else:
        return "Lost Customer"
    
rfm["Segment"] = rfm.apply(segment_customer,axis=1)
print("Customer Segmentation\n")
print(rfm[["customer_id", "Recency", "Frequency", "Monetary", "R_score", "F_score", "M_score", "Segment"]].head(10))
print("\n")

#BUSINESS ANALYSIS
#Customer count 
customer_count = rfm.groupby("Segment")["customer_id"].count()
print("Customer count is:\n",customer_count)
print("\n")

#Customer share
customer_share = (customer_count/customer_count.sum()) * 100
print("Customer share in percentage:\n",customer_share)
print("\n")

#Total Revenue
total_revenue = rfm.groupby("Segment")["Monetary"].sum()
print("Total Revenue:\n",total_revenue)
print("\n")

#Total Revenue Share
total_revenue_share = (total_revenue/total_revenue.sum()) * 100
print("Revenue Share:\n",total_revenue_share)
print("\n")

#Average Order Value  [Average order = total revenue / total order]
total_order = rfm.groupby("Segment")["Frequency"].sum()
print("Total Order is:\n", total_order)
print("\n")

AVG_order_value = total_revenue/total_order
print("Average Order Value is:\n", AVG_order_value)
print("\n")

#Purchase Frequency  [Purchase frequency = total order / total customer]
purchase_frequency = total_order/customer_count
print("Purchase Frequency is:\n", purchase_frequency)
print("\n")

#VISUALIZATION
#Segment distribution
sns.countplot(x="Segment", data=rfm)
plt.title("Segment Distribution")
plt.show()

#Revenue by segment
sns.barplot(x="Segment", y="Monetary", data=rfm, estimator=sum)
plt.title("Revenue by segment")
plt.show()

#Recency vs Frequency scatter plot
sns.scatterplot(x="Recency", y="Frequency", data=rfm)
plt.title("Recency vs Frequency")
plt.show()

#Monetary distribution
sns.histplot(rfm["Monetary"])
plt.title("Monetary distribution")
plt.show()

# #RFM heatmap
# sns.heatmap(rfm)
# plt.title("RFM heatmap")
# plt.show()