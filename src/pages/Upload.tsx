import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertCircle, CheckCircle2, FileSpreadsheet, LogOut, Upload as UploadIcon } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { MappingRequiredError, uploadDataset } from "@/lib/api";
import { ANALYSIS_STORAGE_KEY } from "@/lib/storage";
import type { MappingPrompt } from "@/lib/types";

const MAX_FILE_BYTES = 35 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = [".csv", ".xlsx", ".xls"];

const Upload = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [mappingPrompt, setMappingPrompt] = useState<MappingPrompt | null>(null);
  const [mapping, setMapping] = useState({
    dateColumn: "",
    targetColumn: "",
    itemColumn: "",
  });

  useEffect(() => {
    if (!user) {
      navigate("/");
    }
  }, [user, navigate]);

  const acceptedLabel = useMemo(() => ACCEPTED_EXTENSIONS.join(", "), []);

  const resetMapping = () => {
    setMappingPrompt(null);
    setMapping({
      dateColumn: "",
      targetColumn: "",
      itemColumn: "",
    });
  };

  const handleFile = useCallback((incomingFile: File) => {
    setError("");
    setStatusMessage("");
    resetMapping();

    const extension = `.${incomingFile.name.split(".").pop()?.toLowerCase() ?? ""}`;
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setError(`Please upload one of the supported file types: ${acceptedLabel}`);
      return;
    }

    if (incomingFile.size > MAX_FILE_BYTES) {
      setError("File too large. Maximum supported size is 35 MB.");
      return;
    }

    setFile(incomingFile);
  }, [acceptedLabel]);

  const runAnalysis = useCallback(async (overrides?: { dateColumn?: string; targetColumn?: string; itemColumn?: string }) => {
    if (!file) {
      return;
    }

    setProcessing(true);
    setError("");
    setStatusMessage("Analyzing dataset and preparing dashboard results...");

    try {
      const result = await uploadDataset({
        file,
        dateColumn: overrides?.dateColumn,
        targetColumn: overrides?.targetColumn,
        itemColumn: overrides?.itemColumn,
      });
      localStorage.setItem(ANALYSIS_STORAGE_KEY, JSON.stringify(result));
      localStorage.removeItem("rso_dataset");
      navigate("/dashboard");
    } catch (caught) {
      if (caught instanceof MappingRequiredError) {
        const prompt = caught.prompt;
        setMappingPrompt(prompt);
        setMapping({
          dateColumn: prompt.date_candidates[0] ?? "",
          targetColumn: prompt.target_candidates[0] ?? "",
          itemColumn: prompt.item_candidates[0] ?? "",
        });
        setStatusMessage("The file needs one quick schema confirmation before analysis can continue.");
      } else {
        setError(caught instanceof Error ? caught.message : "Dataset analysis failed.");
        setStatusMessage("");
      }
    } finally {
      setProcessing(false);
    }
  }, [file, navigate]);

  if (!user) {
    return null;
  }

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    if (event.dataTransfer.files[0]) {
      handleFile(event.dataTransfer.files[0]);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <div>
          <h1 className="text-lg font-bold text-foreground">RSO Forecaster</h1>
          <p className="text-xs text-muted-foreground">Upload retail sales data for cleaning, forecasting, and restocking analysis</p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-muted-foreground">{user.name}</span>
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

      <div className="mx-auto mt-16 max-w-3xl px-6 pb-16">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}>
          <h2 className="mb-2 text-2xl font-bold text-foreground">Upload Dataset</h2>
          <p className="mb-8 text-muted-foreground">Upload a CSV or Excel retail sales dataset.</p>

          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`cursor-pointer rounded-xl border-2 border-dashed p-12 text-center transition-all ${
              dragOver
                ? "border-primary bg-primary/5"
                : file
                  ? "border-primary/50 bg-primary/5"
                  : "border-border hover:border-muted-foreground"
            }`}
            onClick={() => {
              const input = document.createElement("input");
              input.type = "file";
              input.accept = ACCEPTED_EXTENSIONS.join(",");
              input.onchange = (event) => {
                const nextFile = (event.target as HTMLInputElement).files?.[0];
                if (nextFile) {
                  handleFile(nextFile);
                }
              };
              input.click();
            }}
          >
            {file ? (
              <div className="flex flex-col items-center gap-3">
                <CheckCircle2 className="h-12 w-12 text-primary" />
                <div>
                  <p className="font-medium text-foreground">{file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB - Ready for backend analysis
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                  <UploadIcon className="h-7 w-7 text-muted-foreground" />
                </div>
                <div>
                  <p className="font-medium text-foreground">Drop your dataset here</p>
                  <p className="text-sm text-muted-foreground">or click to browse</p>
                </div>
                <p className="text-xs text-muted-foreground">Supported: {acceptedLabel} - Max 35 MB</p>
                <FileSpreadsheet className="h-5 w-5 text-muted-foreground/50" />
              </div>
            )}
          </div>

          {statusMessage && !error && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-primary">
              {statusMessage}
            </motion.div>
          )}

          {error && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 flex items-center gap-2 text-sm text-destructive">
              <AlertCircle className="h-4 w-4" />
              {error}
            </motion.div>
          )}

          {mappingPrompt && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mt-6 rounded-xl border border-border bg-card p-6">
              <h3 className="text-sm font-semibold text-foreground">Confirm detected columns once</h3>
              <p className="mt-2 text-sm text-muted-foreground">{mappingPrompt.message}</p>
              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <label className="text-sm text-muted-foreground">
                  Date column
                  <select
                    value={mapping.dateColumn}
                    onChange={(event) => setMapping((current) => ({ ...current, dateColumn: event.target.value }))}
                    className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-foreground outline-none focus:border-primary"
                  >
                    <option value="">Select column</option>
                    {mappingPrompt.columns.map((column) => (
                      <option key={column} value={column}>{column}</option>
                    ))}
                  </select>
                </label>
                <label className="text-sm text-muted-foreground">
                  Target sales column
                  <select
                    value={mapping.targetColumn}
                    onChange={(event) => setMapping((current) => ({ ...current, targetColumn: event.target.value }))}
                    className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-foreground outline-none focus:border-primary"
                  >
                    <option value="">Select column</option>
                    {mappingPrompt.columns.map((column) => (
                      <option key={column} value={column}>{column}</option>
                    ))}
                  </select>
                </label>
                <label className="text-sm text-muted-foreground">
                  Product or item column
                  <select
                    value={mapping.itemColumn}
                    onChange={(event) => setMapping((current) => ({ ...current, itemColumn: event.target.value }))}
                    className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-foreground outline-none focus:border-primary"
                  >
                    <option value="">Select column</option>
                    {mappingPrompt.columns.map((column) => (
                      <option key={column} value={column}>{column}</option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="mt-5 flex flex-wrap gap-3">
                <Button
                  onClick={() => void runAnalysis(mapping)}
                  disabled={processing || !mapping.dateColumn || !mapping.targetColumn}
                >
                  {processing ? "Analyzing..." : "Continue With These Columns"}
                </Button>
                <Button variant="outline" onClick={resetMapping} disabled={processing}>
                  Clear Mapping
                </Button>
              </div>
            </motion.div>
          )}

          {file && !mappingPrompt && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-6">
              <Button className="w-full font-semibold" size="lg" onClick={() => void runAnalysis()} disabled={processing}>
                {processing ? "Analyzing Dataset..." : "Analyze Dataset"}
              </Button>
            </motion.div>
          )}

          <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-3">
            {[
              { label: "CSV + Excel", desc: "Flexible upload support" },
              { label: "Smart Cleaning", desc: "Only applied when needed" },
              { label: "3-Month Forecast", desc: "RF, XGBoost, and LSTM" },
            ].map((item) => (
              <div key={item.label} className="rounded-lg border border-border bg-card p-4 text-center">
                <p className="text-sm font-semibold text-foreground">{item.label}</p>
                <p className="mt-1 text-xs text-muted-foreground">{item.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Upload;
