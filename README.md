# FastFloodEval

This web-app is intended to help users (Individual home buyers or Insurance Companies) to better understand the risk of flooding of the property of interest. Insurance companies can use this tool to accept/mitigate/ or avoid a risk. Home buyers can also decide whether to invest on a property or not.


FEMA flood claims data was preprocessed and used as the underlying data. Machine learning techniques were applied to find the best model that can predict the building loss ratio (amount of claim / insured value).

 ![alt text](static/graph.png)
 
 The best results was achieved with the gradient boosting regressor with R2 score of 67% on the test data.
 
 First user needs to create an account. Then the input information of the building is required. Once input information is entered, you will get to a page where you can see the map of property along with the predicted loss ratio and a box plot showing the historical claim information of the entered ZIP Code (if there be any.)

 
 