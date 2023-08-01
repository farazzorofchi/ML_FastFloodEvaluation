# FastFloodEval

This web-app (https://fast-flood-eval.herokuapp.com/) is intended to help users (Individual home buyers or Insurance Companies) to better understand the risk of flooding of the property of interest. Insurance companies can use this tool to accept/mitigate/avoid a risk. Home buyers can also decide whether to invest on a property or not.

Recently, federal emergency management agency (FEMA) published its flood claim experience data. The dataset represents more than 2 million claim transactions from 1970 to 2019 ([https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims].
FEMA flood claims data was preprocessed and used as the underlying data. Machine learning and deep learning techniques such as XGBRegressor, GradientBoostingRegressor, MLP, Ridge, and Random Forest Regressor were applied to find the best model that can predict the building loss ratio (amount of claim / insured value).

In order to provide a sound approach to quantify the loss ratio (loss / insured value) of buildings with different attributes using historical claim data.

![alt text](static/graph.png)

The best result was achieved with the gradient boosting regressor with R2 score of 67% on the test data.

First user needs to create an account. Then the input information of the building is required. Once input information is entered, you will get to a page where you can see the map of property along with the predicted loss ratio and a box plot showing the historical claim information of the entered ZIP Code (if there is any.)

There are many other attributes that can cause flooding. This tool only uses the historical 
claims data and doesn't guarantee the results to be accurate. We do not take any responsibility regarding the outputs of this tool and how it will be used.
