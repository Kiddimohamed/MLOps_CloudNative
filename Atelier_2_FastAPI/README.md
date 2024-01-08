# FAST API LAB 

FastAPI Provide tools to manage and automate standard tasks when
building web applications
- Session management
- Templating
- Access to databases 
- And other.
In the first section we will use the fourniture dataset and then  the same data that we are working on it in the FLask lab in the second section.
Please run the Furniture notebook and MPI prediction notebook 
## preprocessing
we will start by importing our dataset , before applying preprocessing technic as encoding , scaling and spliting before building our model.

as a models we will compare between , Linear Regression model , Decision tree model,SVR model ,  and Random forest model.
and after comparing the result we find that  the decision tree gives us the better results then we will serialize it to pickle format.

In our main.py , we are starting by Installing joblib , uvicorn, fastapi , and pydantic.

![](https://media.discordapp.net/attachments/1191490101247758479/1193763874009251920/Screenshot_from_2024-01-08_04-51-04.png?ex=65ade63e&is=659b713e&hm=1eb26282f3f3a5fb65ed292f569c8bd7f886a9d72c63cdd0b2e349cde2db588d&=&format=webp&quality=lossless&width=794&height=660)

so In the second section we will try to  pre_processe our data , and build some models before compare and choose the best model and serve it using FastAPI
we will use POSTMANE to test our API .
![](https://media.discordapp.net/attachments/1191490101247758479/1193760750578176030/Screenshot_from_2024-01-08_04-38-45.png?ex=65ade355&is=659b6e55&hm=d9cec6bbab56c47b49609fc86505d034ff6f1cda6a16ccc58278aac02dcd6430&=&format=webp&quality=lossless&width=718&height=660)


