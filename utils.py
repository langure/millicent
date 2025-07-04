import pandas as pd
import os
import matplotlib.pyplot as plt

def read_excel_file(filepath: str):
    """
    Reads an Excel file and returns the DataFrame.
    Args:
        filepath (str): Path to the Excel file.
    Returns:
        pd.DataFrame: The DataFrame from the Excel file.
    """
    df = pd.read_excel(filepath)
    return df

def save_columns_to_file(df, output_path="results/columns.txt"):
    """
    Saves the column names of the DataFrame to a text file.
    Args:
        df (pd.DataFrame): The DataFrame whose columns to save.
        output_path (str): Path to the output text file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for col in df.columns:
            f.write(f"{col}\n")

def save_head_to_file(df, output_path="results/head.txt", n=5):
    """
    Saves the first n rows of the DataFrame to a text file.
    Args:
        df (pd.DataFrame): The DataFrame whose head to save.
        output_path (str): Path to the output text file.
        n (int): Number of rows to save from the top. Default is 5.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.head(n).to_string(open(output_path, "w"), index=False)

def descriptive_statistics(df):
    """
    Prints and saves descriptive statistics of the DataFrame with explanations.
    """
    header = (
        "Descriptive Statistics\n"
        "----------------------\n"
        "This section provides summary statistics for each column, including count, mean, std, min, max, and quartiles.\n"
        "For categorical columns, unique values and most frequent value are shown.\n\n"
    )
    stats = df.describe(include='all').to_string()
    summary = "\n\nSummary: "
    if 'Puntaje Obtenido' in df.columns:
        mean_score = df['Puntaje Obtenido'].mean()
        summary += f"The average score (Puntaje Obtenido) is {mean_score:.2f}. "
        min_score = df['Puntaje Obtenido'].min()
        max_score = df['Puntaje Obtenido'].max()
        summary += f"Scores range from {min_score:.2f} to {max_score:.2f}. "
    summary += "See above for more details on each column."
    print(header + stats + summary)
    with open("results/descriptive_statistics.txt", "w") as f:
        f.write(header + stats + summary)

def grouped_analysis(df):
    """
    Prints and saves grouped analysis by exam type, gender, and location with explanations.
    """
    header = (
        "Grouped Analysis\n"
        "---------------\n"
        "This section shows average 'Puntaje Obtenido' grouped by exam type (TIPO_EXA), gender (SEXO), and location (NOM_SEDE).\n\n"
    )
    output = []
    output.append("Average scores by TIPO_EXA:")
    if 'TIPO_EXA' in df.columns and 'Puntaje Obtenido' in df.columns:
        output.append(df.groupby('TIPO_EXA')['Puntaje Obtenido'].mean().to_string())
    output.append("\nAverage scores by SEXO:")
    if 'SEXO' in df.columns and 'Puntaje Obtenido' in df.columns:
        output.append(df.groupby('SEXO')['Puntaje Obtenido'].mean().to_string())
    output.append("\nAverage scores by NOM_SEDE:")
    if 'NOM_SEDE' in df.columns and 'Puntaje Obtenido' in df.columns:
        output.append(df.groupby('NOM_SEDE')['Puntaje Obtenido'].mean().sort_values(ascending=False).to_string())
    result = "\n".join(output)
    summary = "\n\nSummary: "
    if 'TIPO_EXA' in df.columns and 'Puntaje Obtenido' in df.columns:
        tipo_means = df.groupby('TIPO_EXA')['Puntaje Obtenido'].mean()
        summary += f"Highest average score by exam type: {tipo_means.idxmax()} ({tipo_means.max():.2f}). "
    if 'SEXO' in df.columns and 'Puntaje Obtenido' in df.columns:
        sexo_means = df.groupby('SEXO')['Puntaje Obtenido'].mean()
        summary += f"Highest average score by gender: {sexo_means.idxmax()} ({sexo_means.max():.2f}). "
    if 'NOM_SEDE' in df.columns and 'Puntaje Obtenido' in df.columns:
        sede_means = df.groupby('NOM_SEDE')['Puntaje Obtenido'].mean()
        summary += f"Campus with highest average score: {sede_means.idxmax()} ({sede_means.max():.2f}). "
    print(header + result + summary)
    with open("results/grouped_analysis.txt", "w") as f:
        f.write(header + result + summary)

def frequency_counts(df):
    """
    Prints and saves frequency counts for exam type, location, gender, and DICTAMEN with explanations.
    """
    header = (
        "Frequency Counts\n"
        "----------------\n"
        "This section shows the count of records for key categorical columns.\n\n"
    )
    output = []
    for col in ['TIPO_EXA', 'NOM_SEDE', 'SEXO', 'DICTAMEN']:
        if col in df.columns:
            output.append(f"Counts for {col} (sorted):")
            output.append(df[col].value_counts().sort_values(ascending=False).to_string())
            output.append(f"Total: {df[col].count()}")
            output.append("")
    result = "\n".join(output)
    summary = "\n\nSummary: "
    if 'DICTAMEN' in df.columns:
        most_common = df['DICTAMEN'].value_counts().idxmax()
        count = df['DICTAMEN'].value_counts().max()
        summary += f"Most common DICTAMEN: {most_common} ({count} records). "
    print(header + result + summary)
    with open("results/frequency_counts.txt", "w") as f:
        f.write(header + result + summary)

def correlation_analysis(df):
    """
    Prints and saves correlation matrix for numeric columns with explanations.
    """
    header = (
        "Correlation Analysis\n"
        "--------------------\n"
        "This section shows the correlation matrix for numeric columns. Values close to 1 or -1 indicate strong relationships.\n\n"
    )
    corr = df.corr(numeric_only=True)
    result = corr.to_string()
    summary = "\n\nSummary: "
    corr = df.corr(numeric_only=True)
    if not corr.empty:
        max_corr = corr.where(~(corr == 1)).abs().max().max()
        summary += f"Strongest correlation (excluding self-correlation) is {max_corr:.2f}. "
    print(header + result + summary)
    with open("results/correlation_analysis.txt", "w") as f:
        f.write(header + result + summary)

def trends_and_outliers(df):
    """
    Identifies and saves outliers and trends in scores with explanations.
    """
    header = (
        "Trends and Outliers\n"
        "-------------------\n"
        "This section identifies outliers in 'Puntaje Obtenido' using the IQR method. Outliers are values outside 1.5*IQR from Q1 or Q3.\n\n"
    )
    output = []
    if 'Puntaje Obtenido' in df.columns:
        scores = df['Puntaje Obtenido']
        q1 = scores.quantile(0.25)
        q3 = scores.quantile(0.75)
        iqr = q3 - q1
        outliers = df[(scores < q1 - 1.5 * iqr) | (scores > q3 + 1.5 * iqr)]
        output.append(f"Number of outliers in 'Puntaje Obtenido': {len(outliers)}")
        output.append("Outlier rows (first 5):")
        output.append(outliers.head().to_string())
    else:
        output.append("'Puntaje Obtenido' column not found.")
    result = "\n".join(output)
    summary = "\n\nSummary: "
    if 'Puntaje Obtenido' in df.columns:
        scores = df['Puntaje Obtenido']
        q1 = scores.quantile(0.25)
        q3 = scores.quantile(0.75)
        iqr = q3 - q1
        outliers = df[(scores < q1 - 1.5 * iqr) | (scores > q3 + 1.5 * iqr)]
        summary += f"There are {len(outliers)} outliers in the scores. "
    print(header + result + summary)
    with open("results/trends_and_outliers.txt", "w") as f:
        f.write(header + result + summary)

def missing_data_analysis(df):
    """
    Prints and saves missing data analysis with explanations.
    """
    header = (
        "Missing Data Analysis\n"
        "---------------------\n"
        "This section shows the number of missing values per column.\n\n"
    )
    missing = df.isnull().sum()
    result = missing[missing > 0].to_string()
    if not result or result == '':
        result = "No missing data found."
    summary = "\n\nSummary: "
    if result == "No missing data found.":
        summary += "All columns are complete."
    else:
        summary += "Some columns have missing values."
    print(header + result + summary)
    with open("results/missing_data_analysis.txt", "w") as f:
        f.write(header + result + summary)

def visualizations(df):
    """
    Generates and saves basic visualizations as PNG files, with a summary text file.
    """
    os.makedirs("results", exist_ok=True)
    summary = [
        "Visualizations\n--------------\n",
        "This section lists the generated plots. See the PNG files in the results/ directory.\n"
    ]
    # Histogram of scores
    if 'Puntaje Obtenido' in df.columns:
        plt.figure(figsize=(8, 5))
        df['Puntaje Obtenido'].hist(bins=30)
        plt.title('Histogram of Puntaje Obtenido')
        plt.xlabel('Puntaje Obtenido')
        plt.ylabel('Frequency')
        plt.savefig('results/histogram_puntaje_obtenido.png')
        plt.close()
        summary.append("- histogram_puntaje_obtenido.png: Distribution of scores.\n")
    # Bar chart for DICTAMEN
    if 'DICTAMEN' in df.columns:
        plt.figure(figsize=(8, 5))
        df['DICTAMEN'].value_counts().plot(kind='bar')
        plt.title('Counts by DICTAMEN')
        plt.xlabel('DICTAMEN')
        plt.ylabel('Count')
        plt.savefig('results/bar_dictamen.png')
        plt.close()
        summary.append("- bar_dictamen.png: Count of each DICTAMEN category.\n")
    # Boxplot for scores by SEXO
    if 'SEXO' in df.columns and 'Puntaje Obtenido' in df.columns:
        plt.figure(figsize=(8, 5))
        df.boxplot(column='Puntaje Obtenido', by='SEXO')
        plt.title('Boxplot of Puntaje Obtenido by SEXO')
        plt.suptitle('')
        plt.xlabel('SEXO')
        plt.ylabel('Puntaje Obtenido')
        plt.savefig('results/boxplot_puntaje_by_sexo.png')
        plt.close()
        summary.append("- boxplot_puntaje_by_sexo.png: Score distribution by gender.\n")
    print("".join(summary))
    with open("results/visualizations.txt", "w") as f:
        f.writelines(summary)

def generate_profiling_report(df, output_path="results/profiling_report.html"):
    """
    Generates an automated profiling report using ydata-profiling and saves it as HTML.
    """
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, title="Automated Data Profiling Report", explorative=True)
        profile.to_file(output_path)
        print(f"Profiling report saved to {output_path}")
    except ImportError:
        print("ydata-profiling is not installed. Run 'pip install ydata-profiling' to use this feature.")