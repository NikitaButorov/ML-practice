#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here


    errors = []
    for pred, actual in zip(predictions, net_worths):
        errors.append((pred - actual) ** 2)


    data = []
    for age, net, err in zip(ages, net_worths, errors):
        data.append((age, net, err))


    data_sorted = sorted(data, key=lambda x: x[2])


    cutoff = int(len(data_sorted) * 0.9)
    cleaned_data = data_sorted[:cutoff]

    return cleaned_data

