import pandas as pd


def get_prop(i, result_df):
    try:
        return result_df.iloc[i]["proportion"]
    except (IndexError, KeyError):
        return 0
    

def compute_pronunciation(df: pd.DataFrame):

    if df is None or df.empty:
        raise ValueError("Empty transcript DataFrame")

    # Ensure correct type
    result_df = df.copy()

    
    bins = [0, 0.5, 0.7, 0.85, 0.95, 1.0]
    labels = ["0_50%", "50_70%", "70_85%", "85_95%", "95_100%"]

    result_df["conf_bin"] = pd.cut(result_df["probability"], bins=bins, labels=labels, right=False)

    result_df = result_df.groupby(['conf_bin']).agg({'probability': 'count'}).reset_index()
    result_df['proportion'] = result_df['probability']/result_df['probability'].sum()*100
    
    return {
        "0_50%": get_prop(0, result_df),
        "50_70%": get_prop(1, result_df),
        "70_85%": get_prop(2, result_df),
        "85_95%": get_prop(3, result_df),
        "95_100%": get_prop(4, result_df),
    }