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
    visualizations,
    exploratory_scatter_plots,
    plot_grades_by_state,
    plot_grade_histogram_by_state,
    plot_top3_states_per_grade
)
from regression import linear_regression, polynomial_regression, random_forest_regression, nn_regression

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

    print("\n" + "-"*60 + "\nExploratory Scatter Plots\n" + "-"*60)
    exploratory_scatter_plots(df)

    print("\n" + "-"*60 + "\nLinear Regression\n" + "-"*60)
    linear_regression()

    print("\n" + "-"*60 + "\nPolynomial Regression (degree 3)\n" + "-"*60)
    #polynomial_regression()

    print("\n" + "-"*60 + "\nRandom Forest Regression\n" + "-"*60)
    random_forest_regression()

    print("\n" + "-"*60 + "\nNeural Network Regression (PyTorch)\n" + "-"*60)
    nn_regression()

    print("\n" + "-"*60 + "\nGrades by State (EDO_SEDE)\n" + "-"*60)
    plot_grades_by_state(df)

    print("\n" + "-"*60 + "\nGrade Histogram by State (EDO_SEDE)\n" + "-"*60)
    plot_grade_histogram_by_state(df)

    print("\n" + "-"*60 + "\nTop 3 States per Grade\n" + "-"*60)
    plot_top3_states_per_grade(df)