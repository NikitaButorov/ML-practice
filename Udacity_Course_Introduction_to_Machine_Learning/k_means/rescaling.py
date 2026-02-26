def featureScaling(arr):
    xmin = float(min(arr))
    xmax = float(max(arr))
    newdata = []
    for i in arr:
        x = float((i - xmin) / (xmax - xmin))
        newdata.append(x)

    return newdata


# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)