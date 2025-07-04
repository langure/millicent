from utils import (
    read_excel_file,
    save_columns_to_file,
    save_head_to_file,
    descriptive_statistics,
    grouped_analysis,
    frequency_counts,
    correlation_analysis,
    trends_and_outliers,
    missing_data_analysis,
    visualizations
)

if __name__ == "__main__":
    df = read_excel_file("data/data.xlsx")
    save_columns_to_file(df)
    save_head_to_file(df)

    print("\n" + "-"*60 + "\nDescriptive Statistics\n" + "-"*60)
    descriptive_statistics(df)

    print("\n" + "-"*60 + "\nGrouped Analysis\n" + "-"*60)
    grouped_analysis(df)

    print("\n" + "-"*60 + "\nFrequency Counts\n" + "-"*60)
    frequency_counts(df)

    print("\n" + "-"*60 + "\nCorrelation Analysis\n" + "-"*60)
    correlation_analysis(df)

    print("\n" + "-"*60 + "\nTrends and Outliers\n" + "-"*60)
    trends_and_outliers(df)

    print("\n" + "-"*60 + "\nMissing Data Analysis\n" + "-"*60)
    missing_data_analysis(df)

    print("\n" + "-"*60 + "\nVisualizations\n" + "-"*60)
    visualizations(df)