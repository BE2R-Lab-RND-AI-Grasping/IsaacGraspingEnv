

def resolve_names_matching(names_list: list[str], name_mapping_dict: dict[str, str]):
    """Create a mapping from old names to new names based on a provided dictionary of replacements substrings.
    
    Example:

        names_list = ["003_cracker_box", "004_sugar_box", "005_tomato_soup_can"]
        name_mapping_dict = {"003": "3", "004": "4", "005": "5"}
        d_old2new_names = resolve_names_matching(names_list, name_mapping_dict)
        # d_old2new_names = {"003_cracker_box": "3_cracker_box", 
        # "004_sugar_box": "4_sugar_box", "005_tomato_soup_can": "5_tomato_soup_can"}

    Args:
        names_list (list[str]): _list of names to be converted_
        name_mapping_dict (dict[str, str]): _mapping of old substrings to new substrings_

    Returns:
        dict: _mapping of old names to new names_
    """
    d_old2new_names = {}
    for name in names_list:
        new_name = name + ""
        for old_key, new_key in name_mapping_dict.items():
            if old_key in name:
                new_name = new_name.replace(old_key, new_key, 1)
        d_old2new_names[name] = new_name
        
    return d_old2new_names
