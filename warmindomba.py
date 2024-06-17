import streamlit as st
import csv
import mlxtend.frequent_patterns 
import mlxtend.preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

st.header(":ramen: Warmindo Market Basket Analysis by Rorry!")

st.write("""Di apps ini, anda bisa mencoba customer behaviour pada saat memesan indomie dengan rasa apa. Contoh, kalau konsumen 1 membeli indomie goreng, berapa kemungkinan mereka akan membeli indomie rasa soto betawi""")

st.sidebar.info("Untuk codingnya, bisa ditemukan di my GitHub:")
st.sidebar.link_button("GitHub Source","https://github.com/RorryMiniGunner/PinjemDong/blob/main/06_10_2024_WARMINDO_Market_Basket_Analysis.ipynb")

st.sidebar.warning("Mohon diingat!. Data yang didapat berdasarkan data dari satu warmindo saja (selama Januari - Agustus 2022). Perilaku dari konsumen di tempat lain atau situasi lain, mungkin saja berbeda")

# Read the CSV file into a DataFrame
st.write("Data Warmindo dari Januari - Agustus 2022.")

df = pd.read_csv('warmindo_free_ngulikdata (1).csv')

df['tanggal_transaksi'] = pd.to_datetime(df['tanggal_transaksi'], format='%m/%d/%y').dt.strftime('%d-%m-%Y')
df

#####################

#Indomie Rasa
transaction_list = []

# For loop to create a list of the unique transactions throughout the dataset:
for i in df['tanggal_transaksi'].unique():
    tlist = list(set(df[df['tanggal_transaksi']==i]['nama_produk']))
    if len(tlist)>0:
        transaction_list.append(tlist)
print(len(transaction_list))

te = TransactionEncoder()
te_ary = te.fit(transaction_list).transform(transaction_list)
df2 = pd.DataFrame(te_ary, columns=te.columns_)

#Pembayaran
transaction_list2 = []

for i in df['tanggal_transaksi'].unique():
    tlist2 = list(set(df[df['tanggal_transaksi']==i]['jenis_pembayaran']))
    if len(tlist2)>0:
        transaction_list2.append(tlist2)
print(len(transaction_list2))

te2 = TransactionEncoder()
te_ary2 = te2.fit(transaction_list2).transform(transaction_list2)
df3 = pd.DataFrame(te_ary2, columns=te2.columns_)

frequent_itemsets = apriori(df2, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

frequent_itemsets1 = apriori(df3, min_support=0.01, use_colnames=True)
rules1 = association_rules(frequent_itemsets1, metric='lift', min_threshold=1.0)

#Indomie convert ke string
rules['ant_string'] = rules['antecedents'].apply(lambda x: list(x)[0]).astype("unicode")
rules['con_string'] = rules['consequents'].apply(lambda x: list(x)[0]).astype("unicode")
rules['rule'] = rules['ant_string']+" -> "+rules['con_string']

#Pembayaran convert ke string
rules1['ant_string'] = rules1['antecedents'].apply(lambda x: list(x)[0]).astype("unicode")
rules1['con_string'] = rules1['consequents'].apply(lambda x: list(x)[0]).astype("unicode")
rules1['rule'] = rules1['ant_string']+" -> "+rules1['con_string']

## Sidebar Content
item_filter = st.sidebar.selectbox("Pilih menu rasa indomiemu:",df['nama_produk'].unique())
item_df = rules[rules['ant_string']==item_filter]

item_filter2 = st.sidebar.selectbox("Pilih Pembayaranmu:",df['jenis_pembayaran'].unique())
item_df2 = rules1[rules1['ant_string']==item_filter2]

## Make some notes in the sidebar
st.sidebar.markdown("""### Some Important Terms
**Confidence:** Kemungkinan Konsumen membeli Y, setelah mereka membeli X. Contoh, kalau confidence `Indomie Soto Betawi -> Indomie Soto Padang` is 100% maka kemungkinan konsumen memesan indomie soto betawi pasti memesan indomie soto padang

**Lift:** An expression of the strength of a rule. Contoh Jika X maka Y kemungkinan nilainya > 1, maka probabilitasnya akan sering muncul. Jika X maka Y < 1, maka probabilitas kemunculannya tidak ada.""")

#############################
## Rules Overview Section  ##
#############################

st.subheader("Top 10 Rasa Pilihan Indomie-mu:ramen:")
st.write(" Hasil 10 Teratas Dari Rasa Indomie Yang Dipilih Konsumen di Warmindo ini.")

metrix = pd.DataFrame(rules)
metrix[['antecedents','consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False)
st.write(metrix[['antecedents','consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(10))

st.write("Hasil ini berdasarkan kolerasi antecedents dan consequents, yang mempengaruhi nilai support confidence, confidence, dan lift")

#Data 10 teratas pembayaran di Warmindo

st.subheader("Top 10 Pembayaran di Warmindo:coin:")

metrix1 = pd.DataFrame(rules1)
metrix1[['antecedents','consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False)
st.write(metrix1[['antecedents','consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(10))

st.write(" Hasil 10 Teratas Dari Rasa Indomie Yang Dipilih Konsumen di Warmindo ini.")


#########################
## Single Item Section ##
#########################

st.subheader("Analisa Dari Tiap Produk Indomie Favorit-mu:ramen:")
st.write("Lakukan analisa ini dengan filter bar di sebelah kiri.")

# now you can sort the DataFrame

item_df.sort_values(by='lift', ascending=False, inplace=True)
item_df.reset_index(inplace = True)

# Exclude bad rules (Lift <=1) & write to a table
st.write(item_df[item_df['lift']>1][['rule','confidence','lift']].head(10))

st.write("Analisa ini berdasarkan korelasi X -> Y dan probabilitasnya akan muncul kembali")

###############

st.subheader("Analisa Dari Tiap Jenis Pembayaranmu:coin:")
st.write("Lakukan analisa ini dengan filter bar di sebelah kiri, dibawah filter jenis indomie.")

item_df2.sort_values(by='lift', ascending=False, inplace=True)
item_df2.reset_index(inplace = True)

# Exclude bad rules (Lift <=1) & write to a table
st.write(item_df2[item_df2['lift']>1][['rule','confidence','lift']].head(10))

st.write("Analisa ini berdasarkan korelasi X -> Y dan probabilitasnya akan muncul kembali")

st.subheader("Semoga bisa menjadi bahan pelajaran dan bahasan bagi kalian, mohon maaf jika ada kekurangan, Terima kasih. Mari Makan:ramen:")