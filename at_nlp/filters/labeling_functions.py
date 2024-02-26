from snorkel.labeling import labeling_function, LabelingFunction


@labeling_function()
def string_keyword_strainer(ds: pd.Series) -> int:
    """
    Return a list of random ingredients as strings.
    """
    pred = self.filter_result.ABSTAIN.value
    for keyword in self.keyword_register:
        if ds["Message"].find(keyword) >= 0:
            pred = self.filter_result.RECYCLE.value
            break
    return pred