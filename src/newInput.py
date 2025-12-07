from newPredictions_pipline import predict_sales, load_models

load_models("XGBoost")

# sales_a = predict_sales(store_id=1, date_str="2015-07-31")
# print(f"Store 1 on 2014-05-05 Sales Prediction: ${sales_a:.2f}")
sales_a = predict_sales(store_id=1, date_str="2015-07-31")
print("Store 1 on 2015-07-31 Sales Prediction: $", sales_a, " it supposed to be 5300")
sales_a = predict_sales(store_id=2, date_str="2015-07-31")
print("Store 2 on 2015-07-31 Sales Prediction: $", sales_a, "  it supposed to be 6050")
sales_a = predict_sales(store_id=4, date_str="2015-07-31")
print("Store 4 on 2015-07-31 Sales Prediction: $", sales_a, " it supposed to be 14K")
sales_a = predict_sales(store_id=5, date_str="2015-07-31")
print("Store 1 on 2015-07-31 Sales Prediction: $", sales_a, " it supposed to be 4880")
