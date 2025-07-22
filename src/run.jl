using CSV, DataFrames

include("getting100SampleData.jl")

df1 = CSV.read("Cases/100d_verify.csv", DataFrame)
df_unique = unique(df1)
CSV.write("Cases/100d_verify", df_unique)

df2 = CSV.read("Cases/100d.csv", DataFrame)
isequal(df_unique, df2) ? println("Same") : println("Different")



