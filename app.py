from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request body
class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
def create_item(item: Item):
    return {"name": item.name, "price_with_tax": item.price * 1.17}
