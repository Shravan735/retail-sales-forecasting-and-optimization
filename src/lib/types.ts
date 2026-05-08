export interface SchemaInfo {
  date_column: string | null;
  year_column: string | null;
  month_column: string | null;
  time_column: string | null;
  target_column: string | null;
  item_column: string | null;
  item_group_column: string | null;
  sales_columns: string[];
  date_candidates: string[];
  target_candidates: string[];
  item_candidates: string[];
  can_auto_detect: boolean;
}

export interface CleaningSummary {
  original_rows: number;
  rows_after_cleaning: number;
  duplicates_removed: number;
  invalid_dates_removed: number;
  trimmed_text_columns: string[];
  missing_values_before: Record<string, number>;
  missing_values_after: Record<string, number>;
  filled_numeric_columns: Record<string, number>;
  filled_categorical_columns: Record<string, string>;
  cleaning_applied: boolean;
  original_columns: string[];
  normalized_columns: string[];
}

export interface OverviewInfo {
  row_count: number;
  column_count: number;
  date_range: {
    start: string;
    end: string;
  };
  target_column: string;
  target_sum: number;
  target_mean: number;
  item_count: number | null;
  columns: string[];
}

export interface MonthlyMeta {
  missing_months_filled: number;
  partial_last_period_removed: boolean;
  series_length: number;
  observed_months?: number;
  gap_handling?: string;
}

export interface ConfidenceSummary {
  label: string;
  reason: string;
  beats_baseline: boolean;
  baseline_model: string | null;
  baseline_rmse: number | null;
  best_model_rmse: number | null;
  best_model_r2?: number | null;
  best_model_mape?: number | null;
  sparse_series?: boolean;
}

export interface EdaSegment {
  name: string;
  value: number;
}

export interface EdaMonthPoint {
  month_name: string;
  target: number;
}

export interface EdaSummary {
  trend_direction: string;
  anomaly_count: number;
  top_segments: EdaSegment[];
  seasonal_pattern: EdaMonthPoint[];
  peak_month: {
    date: string;
    value: number;
  };
  lowest_month: {
    date: string;
    value: number;
  };
  average_monthly_sales: number;
  context_columns: Array<{
    type: string;
    columns: string[];
  }>;
}

export interface ModelResult {
  model: string;
  mae?: number | null;
  rmse?: number | null;
  mape?: number | null;
  r2?: number | null;
  status: "trained" | "skipped";
  reason?: string;
}

export interface ForecastPoint {
  date: string;
  value: number;
}

export interface Recommendation {
  item: string;
  group: string;
  recent_sales: number;
  prior_sales: number;
  growth_pct: number;
  warehouse_support: number;
  risk: "low" | "medium" | "high";
  action: string;
  reason: string;
}

export interface AnalysisResult {
  run_id: string;
  file: {
    name: string;
    size_bytes: number;
  };
  schema: SchemaInfo;
  cleaning_summary: CleaningSummary;
  overview: OverviewInfo;
  monthly_meta: MonthlyMeta;
  confidence_summary: ConfidenceSummary;
  eda_summary: EdaSummary;
  monthly_series: ForecastPoint[];
  preview_rows: Record<string, string | number | null>[];
  model_results: ModelResult[];
  best_model: {
    model: string;
    mae?: number | null;
    rmse?: number | null;
    mape?: number | null;
    r2?: number | null;
    reason?: string;
  };
  forecasts: Record<string, ForecastPoint[]>;
  forecast_summary: ForecastPoint[];
  recommendations: Recommendation[];
  generated_at: string;
}

export interface MappingPrompt {
  message: string;
  requires_mapping: true;
  columns: string[];
  date_candidates: string[];
  target_candidates: string[];
  item_candidates: string[];
}

export interface ChatResponse {
  answer: string;
  run_id: string;
}
