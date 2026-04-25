from typing import List
from pydantic import BaseModel, ConfigDict, Field

class ChurnPredictionRequest(BaseModel):
    Gender: str = Field(..., description="Customer's gender (Male, Female)")
    Senior_Citizen: str = Field(..., alias="Senior Citizen", description="Is the customer a senior citizen? (Yes, No)")
    Partner: str = Field(..., description="Does the customer have a partner? (Yes, No)")
    Dependents: str = Field(..., description="Does the customer have dependents? (Yes, No)")
    Tenure_Months: int = Field(..., alias="Tenure Months", description="Number of months the customer has stayed with the company")
    Phone_Service: str = Field(..., alias="Phone Service", description="Does the customer have phone service? (Yes, No)")
    Multiple_Lines: str = Field(..., alias="Multiple Lines", description="Does the customer have multiple lines? (Yes, No, No phone service)")
    Internet_Service: str = Field(..., alias="Internet Service", description="Customer's internet service provider (DSL, Fiber optic, No)")
    Online_Security: str = Field(..., alias="Online Security", description="Does the customer have online security? (Yes, No, No internet service)")
    Online_Backup: str = Field(..., alias="Online Backup", description="Does the customer have online backup? (Yes, No, No internet service)")
    Device_Protection: str = Field(..., alias="Device Protection", description="Does the customer have device protection? (Yes, No, No internet service)")
    Tech_Support: str = Field(..., alias="Tech Support", description="Does the customer have tech support? (Yes, No, No internet service)")
    Streaming_TV: str = Field(..., alias="Streaming TV", description="Does the customer have streaming TV? (Yes, No, No internet service)")
    Streaming_Movies: str = Field(..., alias="Streaming Movies", description="Does the customer have streaming movies? (Yes, No, No internet service)")
    Contract: str = Field(..., description="The contract term of the customer (Month-to-month, One year, Two year)")
    Paperless_Billing: str = Field(..., alias="Paperless Billing", description="Does the customer have paperless billing? (Yes, No)")
    Payment_Method: str = Field(..., alias="Payment Method", description="The customer's payment method")
    Monthly_Charges: float = Field(..., alias="Monthly Charges", description="The amount charged to the customer monthly")
    Total_Charges: float = Field(..., alias="Total Charges", description="The total amount charged to the customer")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Gender": "Male",
                "Senior Citizen": "No",
                "Partner": "No",
                "Dependents": "No",
                "Tenure Months": 2,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "DSL",
                "Online Security": "Yes",
                "Online Backup": "Yes",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Mailed check",
                "Monthly Charges": 53.85,
                "Total Charges": 108.15
            }
        }
    )

class ChurnPredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    churn_probability: float
    risk_level: str

class BatchPredictResponseItem(BaseModel):
    prediction: int
    prediction_label: str
    churn_probability: float
    risk_level: str
