def split_by_criteria(dictionary, keys):  # type: (dict, set) -> (dict, dict)
    """Split a dictionary in two by the provided keys.

    Args:
        dictionary (dict[str, object]): A Python dictionary
        keys (set[str]): Set of keys which will be the split criteria

    Returns:
        criteria (dict[string, object]), not_criteria (dict[string, object]): the result of the split criteria.
    """
    dict_matching_criteria = {k: dictionary[k] for k in dictionary.keys() if k in keys}
    dict_not_matching_criteria = {k: dictionary[k] for k in dictionary.keys() if k not in keys}

    return dict_matching_criteria, dict_not_matching_criteria
