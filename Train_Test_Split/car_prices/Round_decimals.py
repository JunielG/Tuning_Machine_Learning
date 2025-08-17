score = lg_model.score(r_X_test, r_y_test)
formatted_score = f"{score:.4f}"
print(formatted_score)

# Round it
score = lg_model.score(r_X_test, r_y_test)
rounded_score = round(score, 4)
print(rounded_score)