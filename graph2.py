import pandas as pd

# Sample schema with multiple entity types
customers = pd.DataFrame({
    "customer_id": ["C1", "C2", "C3", "C4", "C5"]
})

sellers = pd.DataFrame({
    "seller_id": ["S1", "S2"]
})

brands = pd.DataFrame({
    "brand_id": ["B1", "B2"],
    "brand_name": ["Philips", "GenericCo"]
})

products = pd.DataFrame({
    "product_id": ["P1", "P2", "P3", "P4"],
    "brand_id": ["B1", "B1", "B2", "B2"],
    "seller_id": ["S1", "S1", "S2", "S2"]
})

reviews = pd.DataFrame({
    "review_id": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
    "customer_id": ["C1", "C1", "C2", "C3", "C4", "C5", "C5"],
    "product_id": ["P1", "P2", "P2", "P3", "P4", "P3", "P4"],
    "rating": [5, 5, 4, 1, 1, 5, 5]
})

# Display the entity and relationship tables for the user
import ace_tools as tools; tools.display_dataframe_to_user(name="Customers", dataframe=customers)
tools.display_dataframe_to_user(name="Sellers", dataframe=sellers)
tools.display_dataframe_to_user(name="Brands", dataframe=brands)
tools.display_dataframe_to_user(name="Products", dataframe=products)
tools.display_dataframe_to_user(name="Reviews", dataframe=reviews)
