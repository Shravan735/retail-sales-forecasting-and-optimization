import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  BarChart3,
  Brain,
  Database,
  LineChart,
  LogOut,
  Package,
  RefreshCcw,
  TrendingUp,
  UploadCloud,
  Wand2,
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import ChatbotPanel from "@/components/ChatbotPanel";
import { ANALYSIS_STORAGE_KEY } from "@/lib/storage";
import type { AnalysisResult, ForecastPoint, ModelResult } from "@/lib/types";

interface StatCardProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  sub?: string;
}

const StatCard = ({ icon: Icon, label, value, sub }: StatCardProps) => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-xl border border-border bg-card p-5">
    <div className="mb-2 flex items-center gap-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
        <Icon className="h-4 w-4 text-primary" />
      </div>
      <span className="text-xs uppercase tracking-wider text-muted-foreground">{label}</span>
    </div>
    <p className="text-2xl font-bold text-foreground">{value}</p>
    {sub && <p className="mt-1 text-xs text-muted-foreground">{sub}</p>}
  </motion.div>
);

const formatCompactNumber = (value: number) =>
  new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);

const formatMetric = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(2);
};

const getMonthlyValue = (point: { value?: number; target?: number }) => point.value ?? point.target ?? 0;

const chartTooltipStyle = {
  backgroundColor: "hsl(220, 18%, 10%)",
  border: "1px solid hsl(220, 14%, 18%)",
  borderRadius: "8px",
  fontSize: 12,
};

const readAnalysisResult = (): AnalysisResult | null => {
  try {
    const raw = localStorage.getItem(ANALYSIS_STORAGE_KEY);
    if (!raw) {
      return null;
    }

    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "run_id" in parsed &&
      "overview" in parsed &&
      "monthly_series" in parsed
    ) {
      return parsed as AnalysisResult;
    }
  } catch {
    return null;
  }

  return null;
};

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    if (!user) {
      navigate("/");
      return;
    }

    const stored = readAnalysisResult();
    if (!stored) {
      navigate("/upload");
      return;
    }

    setAnalysis(stored);
  }, [user, navigate]);

  const trainedModels = useMemo(() => {
    return (analysis?.model_results ?? []).filter(
      (result): result is ModelResult => result.status === "trained",
    );
  }, [analysis]);

  const trendData = useMemo(() => {
    if (!analysis) {
      return [] as Array<{ date: string; historical?: number; forecast?: number; divider?: number }>;
    }

    const history = analysis.monthly_series.map((point) => ({
      date: point.date.slice(0, 7),
      historical: getMonthlyValue(point),
      forecast: undefined,
    }));

    if (!analysis.forecast_summary.length) {
      return history;
    }

    if (history.length) {
      const lastHistoryPoint = history[history.length - 1];
      history[history.length - 1] = {
        ...lastHistoryPoint,
        forecast: lastHistoryPoint.historical,
      };
    }

    const forecast = analysis.forecast_summary.map((point) => ({
      date: point.date.slice(0, 7),
      historical: undefined,
      forecast: point.value,
    }));

    return [...history, ...forecast];
  }, [analysis]);

  const seasonalData = useMemo(() => analysis?.eda_summary.seasonal_pattern ?? [], [analysis]);
  const segmentData = useMemo(() => (analysis?.eda_summary.top_segments ?? []).slice(0, 5), [analysis]);

  const modelTable = useMemo(() => {
    return trainedModels.map((result) => ({
      model: result.model,
      mae: formatMetric(result.mae),
      rmse: formatMetric(result.rmse),
      mape: formatMetric(result.mape),
      r2: formatMetric(result.r2),
      selected: result.model === analysis?.best_model.model,
    }));
  }, [trainedModels, analysis]);

  const cleaningItems = useMemo(() => {
    if (!analysis) {
      return [] as Array<{ label: string; value: string }>;
    }

    const summary = analysis.cleaning_summary;
    return [
      { label: "Rows before cleaning", value: summary.original_rows.toLocaleString() },
      { label: "Rows after cleaning", value: summary.rows_after_cleaning.toLocaleString() },
      { label: "Duplicates removed", value: summary.duplicates_removed.toLocaleString() },
      { label: "Invalid dates removed", value: summary.invalid_dates_removed.toLocaleString() },
      { label: "Invalid target rows removed", value: (summary.target_values_removed ?? 0).toLocaleString() },
      { label: "Monthly gaps handled", value: `${analysis.monthly_meta.missing_months_filled.toLocaleString()} (${analysis.monthly_meta.missing_months_method ?? "not needed"})` },
      { label: "Trimmed text columns", value: summary.trimmed_text_columns.length ? summary.trimmed_text_columns.join(", ") : "None" },
    ];
  }, [analysis]);

  if (!analysis) {
    return null;
  }

  const bestModelName = analysis.best_model.model || "Unavailable";
  const topRecommendation = analysis.recommendations[0];
  const previewColumns = Object.keys(analysis.preview_rows[0] ?? {}).slice(0, 6);
  const forecastAverage = analysis.forecast_summary.length
    ? analysis.forecast_summary.reduce((sum, point) => sum + point.value, 0) / analysis.forecast_summary.length
    : 0;
  const recommendationHeadline = topRecommendation
    ? topRecommendation.risk === "low"
      ? `${topRecommendation.item} currently has the strongest recent demand.`
      : `${topRecommendation.item} needs inventory attention.`
    : "Recommendations ready";
  const contextInfo = analysis.eda_summary.context_columns.length
    ? analysis.eda_summary.context_columns.map((item) => `${item.type}: ${item.columns.join(", ")}`).join(" | ")
    : "No extra promotion, holiday, or store columns were detected in this upload.";
  const forecastStartLabel = analysis.forecast_summary[0]?.date.slice(0, 7) ?? null;

  return (
    <div className="min-h-screen bg-background">
      <div className="sticky top-0 z-50 flex items-center justify-between border-b border-border bg-background/80 px-6 py-4 backdrop-blur-md">
        <div>
          <h1 className="text-lg font-bold text-foreground">RSO Dashboard</h1>
          <p className="text-xs text-muted-foreground">{analysis.file.name}</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={() => navigate("/upload")}>
            <UploadCloud className="mr-1 h-4 w-4" /> New Upload
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              logout();
              navigate("/");
            }}
          >
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="mx-auto max-w-7xl space-y-6 p-6">
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <StatCard icon={Database} label="Rows" value={analysis.overview.row_count.toLocaleString()} sub={`${analysis.overview.column_count} columns detected`} />
          <StatCard icon={Wand2} label="Cleaning" value={analysis.cleaning_summary.cleaning_applied ? "Applied" : "Minimal"} sub={`${analysis.cleaning_summary.duplicates_removed} duplicates removed`} />
          <StatCard icon={Brain} label="Selected Model" value={bestModelName} sub={`Reliability ${analysis.confidence_summary.label}`} />
          <StatCard icon={TrendingUp} label="Forecast Horizon" value={`${analysis.forecast_summary.length} Months`} sub={recommendationHeadline} />
        </div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-xl border border-border bg-card p-6">
          <h3 className="text-sm font-semibold text-foreground">Plain-Language Summary</h3>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <div className="rounded-lg bg-muted/40 p-4">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">Forecast average</p>
              <p className="mt-2 text-xl font-semibold text-foreground">{formatCompactNumber(forecastAverage)}</p>
              <p className="mt-1 text-xs text-muted-foreground">Average predicted monthly value for the next 3 months.</p>
            </div>
            <div className="rounded-lg bg-muted/40 p-4">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">Forecast reliability</p>
              <p className="mt-2 text-xl font-semibold text-foreground">{analysis.confidence_summary.label}</p>
              <p className="mt-1 text-xs text-muted-foreground">{analysis.confidence_summary.reason}</p>
            </div>
            <div className="rounded-lg bg-muted/40 p-4">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">Business takeaway</p>
              <p className="mt-2 text-sm font-semibold text-foreground">{recommendationHeadline}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {topRecommendation ? topRecommendation.action : "Upload a richer item-level dataset for stronger recommendations."}
              </p>
            </div>
          </div>
        </motion.div>

        <div className="grid gap-6 xl:grid-cols-[1.8fr_1fr]">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-xl border border-border bg-card p-6">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <h3 className="text-sm font-semibold text-foreground">Sales Trend and Forecast</h3>
                <p className="mt-1 text-xs text-muted-foreground">Historical monthly {analysis.overview.target_column} values followed by the next 3 predicted months.</p>
              </div>
              <div className="rounded-lg bg-muted px-3 py-2 text-right">
                <p className="text-xs text-muted-foreground">Target total</p>
                <p className="text-sm font-semibold text-foreground">{formatCompactNumber(analysis.overview.target_sum)}</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 14%, 18%)" />
                <XAxis dataKey="date" stroke="hsl(215, 12%, 50%)" fontSize={11} />
                <YAxis stroke="hsl(215, 12%, 50%)" fontSize={11} />
                <Tooltip
                  cursor={false}
                  contentStyle={chartTooltipStyle}
                />
                {forecastStartLabel ? (
                  <ReferenceLine x={forecastStartLabel} stroke="hsl(40, 90%, 55%)" strokeDasharray="6 6" />
                ) : null}
                <Area type="monotone" dataKey="historical" name="Historical sales" stroke="hsl(160, 84%, 39%)" fill="hsla(160, 84%, 39%, 0.18)" strokeWidth={2} />
                <Area type="monotone" dataKey="forecast" name="Forecasted sales" stroke="hsl(40, 90%, 55%)" fill="hsla(40, 90%, 55%, 0.12)" strokeWidth={2.5} />
              </AreaChart>
            </ResponsiveContainer>
            <div className="mt-3 rounded-lg bg-muted/40 px-4 py-3 text-xs text-muted-foreground">
              Green shows observed sales history. Orange begins at the dotted divider and shows the next 3 forecasted months.
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.08 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-1 text-sm font-semibold text-foreground">Detected Dataset Structure</h3>
            <p className="mb-4 text-xs text-muted-foreground">What the system identified from the uploaded file.</p>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Detected target</span>
                <span className="text-right font-medium text-foreground">{analysis.schema.target_column ?? "Not detected"}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Date logic</span>
                <span className="text-right font-medium text-foreground">{analysis.schema.date_column ?? `${analysis.schema.year_column ?? "year"} + ${analysis.schema.month_column ?? "month"}`}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Item field</span>
                <span className="text-right font-medium text-foreground">{analysis.schema.item_column ?? "Not detected"}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Date range</span>
                <span className="text-right font-medium text-foreground">{analysis.overview.date_range.start} to {analysis.overview.date_range.end}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Unique items</span>
                <span className="text-right font-medium text-foreground">{analysis.overview.item_count?.toLocaleString() ?? "Not available"}</span>
              </div>
            </div>
            <div className="mt-4 rounded-lg bg-muted/40 p-3 text-xs text-muted-foreground">{contextInfo}</div>
          </motion.div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.12 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-foreground">
              <LineChart className="h-4 w-4 text-primary" />
              Exploratory Data Insights
            </h3>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg bg-muted/40 p-4">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Trend direction</p>
                <p className="mt-2 text-lg font-semibold text-foreground">{analysis.eda_summary.trend_direction}</p>
              </div>
              <div className="rounded-lg bg-muted/40 p-4">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Average monthly sales</p>
                <p className="mt-2 text-lg font-semibold text-foreground">{formatCompactNumber(analysis.eda_summary.average_monthly_sales)}</p>
              </div>
              <div className="rounded-lg bg-muted/40 p-4">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Monthly anomalies</p>
                <p className="mt-2 text-lg font-semibold text-foreground">{analysis.eda_summary.anomaly_count}</p>
              </div>
            </div>
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="rounded-lg bg-muted/40 p-4 text-sm">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Peak month</p>
                <p className="mt-2 font-semibold text-foreground">{analysis.eda_summary.peak_month.date}</p>
                <p className="text-muted-foreground">{formatCompactNumber(analysis.eda_summary.peak_month.value)}</p>
              </div>
              <div className="rounded-lg bg-muted/40 p-4 text-sm">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Lowest month</p>
                <p className="mt-2 font-semibold text-foreground">{analysis.eda_summary.lowest_month.date}</p>
                <p className="text-muted-foreground">{formatCompactNumber(analysis.eda_summary.lowest_month.value)}</p>
              </div>
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.16 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-4 text-sm font-semibold text-foreground">Top Sales Segments</h3>
            {segmentData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={segmentData} layout="vertical" margin={{ left: 10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 14%, 18%)" />
                  <XAxis type="number" stroke="hsl(215, 12%, 50%)" fontSize={11} />
                  <YAxis dataKey="name" type="category" stroke="hsl(215, 12%, 50%)" fontSize={11} width={100} />
                  <Tooltip
                    cursor={false}
                    contentStyle={chartTooltipStyle}
                  />
                  <Bar
                    dataKey="value"
                    name="Sales"
                    fill="hsl(160, 84%, 39%)"
                    radius={[0, 4, 4, 0]}
                    activeBar={{ fill: "hsl(160, 84%, 52%)", stroke: "hsl(160, 84%, 64%)", strokeWidth: 1 }}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="rounded-lg bg-muted/40 p-4 text-sm text-muted-foreground">This upload did not include a usable item or category field for segment-level EDA.</div>
            )}
          </motion.div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.18 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-4 text-sm font-semibold text-foreground">Seasonality Pattern</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={seasonalData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 14%, 18%)" />
                <XAxis dataKey="month_name" stroke="hsl(215, 12%, 50%)" fontSize={11} />
                <YAxis stroke="hsl(215, 12%, 50%)" fontSize={11} />
                <Tooltip
                  cursor={false}
                  contentStyle={chartTooltipStyle}
                />
                <Bar
                  dataKey="target"
                  name="Average monthly sales"
                  fill="hsl(200, 80%, 50%)"
                  radius={[4, 4, 0, 0]}
                  activeBar={{ fill: "hsl(200, 90%, 62%)", stroke: "hsl(200, 90%, 72%)", strokeWidth: 1 }}
                />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.22 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-4 text-sm font-semibold text-foreground">Model Evaluation</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-border text-xs uppercase tracking-wider text-muted-foreground">
                    <th className="px-3 py-2 font-medium">Model</th>
                    <th className="px-3 py-2 font-medium">MAE</th>
                    <th className="px-3 py-2 font-medium">RMSE</th>
                    <th className="px-3 py-2 font-medium">MAPE</th>
                    <th className="px-3 py-2 font-medium">R2</th>
                  </tr>
                </thead>
                <tbody>
                  {modelTable.map((row) => (
                    <tr key={row.model} className={`border-b border-border/60 last:border-b-0 ${row.selected ? "bg-primary/5" : ""}`}>
                      <td className="px-3 py-2 text-foreground">{row.model}{row.selected ? " (Selected)" : ""}</td>
                      <td className="px-3 py-2 text-foreground">{row.mae}</td>
                      <td className="px-3 py-2 text-foreground">{row.rmse}</td>
                      <td className="px-3 py-2 text-foreground">{row.mape}</td>
                      <td className="px-3 py-2 text-foreground">{row.r2}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 space-y-3">
              <div className="rounded-lg bg-muted/40 p-4 text-xs text-muted-foreground">
                Lower MAE, RMSE, and MAPE values are better. Higher R2 is better. The selected model is the candidate with the strongest leakage-free holdout RMSE, including the seasonal baseline when it wins.
              </div>
              <div className="rounded-lg bg-primary/5 p-4 text-sm text-foreground">
                <span className="font-semibold">Why {bestModelName} was selected:</span> {analysis.best_model.reason ?? "It gave the most stable result among the trained models for this upload."}
              </div>
            </div>
          </motion.div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.26 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-foreground">
              <RefreshCcw className="h-4 w-4 text-primary" />
              Cleaning Summary
            </h3>
            <div className="space-y-3">
              {cleaningItems.map((item) => (
                <div key={item.label} className="flex flex-col gap-1 rounded-lg bg-muted/40 p-3 text-sm">
                  <span className="text-xs uppercase tracking-wider text-muted-foreground">{item.label}</span>
                  <span className="break-words text-foreground">{item.value}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 rounded-lg bg-muted/40 p-4 text-xs text-muted-foreground">
              Monthly gap handling: {analysis.monthly_meta.gap_handling ?? "No extra monthly-gap handling was required."}
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-foreground">
              <Package className="h-4 w-4 text-primary" />
              Inventory Recommendations
            </h3>
            {analysis.recommendations.length > 0 ? (
              <div className="space-y-3">
                {analysis.recommendations.map((recommendation) => (
                  <div key={recommendation.item} className="flex flex-col gap-3 rounded-lg bg-muted/40 p-4 md:flex-row md:items-center md:justify-between">
                    <div>
                      <p className="text-sm font-medium text-foreground">{recommendation.item}</p>
                      <p className="text-xs text-muted-foreground">{recommendation.group} - {recommendation.reason}</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Recent sales {formatCompactNumber(recommendation.recent_sales)} vs prior {formatCompactNumber(recommendation.prior_sales)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-semibold text-foreground">{recommendation.action}</p>
                      <span
                        className={`mt-2 inline-flex rounded-full px-2 py-1 text-xs ${
                          recommendation.risk === "low"
                            ? "bg-primary/20 text-primary"
                            : recommendation.risk === "medium"
                              ? "bg-yellow-500/20 text-yellow-400"
                              : "bg-destructive/20 text-destructive"
                        }`}
                      >
                        {recommendation.risk} risk
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg bg-muted/40 p-4 text-sm text-muted-foreground">
                The upload was analyzed successfully, but the data is not rich enough for stronger item-level stock guidance.
              </div>
            )}
          </motion.div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.34 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-foreground">
              <BarChart3 className="h-4 w-4 text-primary" />
              Forecast Output
            </h3>
            <div className="space-y-3">
              {analysis.forecast_summary.map((point: ForecastPoint) => (
                <div key={point.date} className="flex items-center justify-between rounded-lg bg-muted/40 p-3 text-sm">
                  <span className="text-muted-foreground">{point.date}</span>
                  <span className="font-semibold text-foreground">{formatCompactNumber(point.value)}</span>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.38 }} className="rounded-xl border border-border bg-card p-6">
            <h3 className="mb-4 text-sm font-semibold text-foreground">Cleaned Data Preview</h3>
            {analysis.preview_rows.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-border text-xs uppercase tracking-wider text-muted-foreground">
                      {previewColumns.map((column) => (
                        <th key={column} className="px-3 py-2 font-medium">{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.preview_rows.slice(0, 8).map((row, rowIndex) => (
                      <tr key={rowIndex} className="border-b border-border/60 last:border-b-0">
                        {previewColumns.map((column) => (
                          <td key={`${rowIndex}-${column}`} className="max-w-[180px] truncate px-3 py-2 text-foreground">
                            {String(row[column] ?? "-")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="rounded-lg bg-muted/40 p-4 text-sm text-muted-foreground">Preview rows are not available for this run.</div>
            )}
          </motion.div>
        </div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.42 }}>
          <ChatbotPanel runId={analysis.run_id} bestModelName={bestModelName} confidenceLabel={analysis.confidence_summary.label} />
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
