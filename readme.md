docker exec -it test_flask_1 python train_model.py



curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"flower":"1,2,3,4"}' \
  http://localhost:5000/iris_post
  