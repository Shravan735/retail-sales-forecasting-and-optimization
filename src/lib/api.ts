import type { AnalysisResult, ChatResponse, MappingPrompt } from "@/lib/types";

const API_BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "";

const buildUrl = (path: string) => `${API_BASE}${path}`;

const isMappingPrompt = (value: unknown): value is MappingPrompt => {
  return (
    typeof value === "object" &&
    value !== null &&
    "requires_mapping" in value &&
    (value as MappingPrompt).requires_mapping === true
  );
};

export class MappingRequiredError extends Error {
  prompt: MappingPrompt;

  constructor(prompt: MappingPrompt) {
    super(prompt.message);
    this.name = "MappingRequiredError";
    this.prompt = prompt;
  }
}

const readErrorResponse = async (response: Response): Promise<never> => {
  let payload: unknown = null;

  try {
    payload = await response.json();
  } catch {
    throw new Error(`Request failed with status ${response.status}`);
  }

  const detail =
    typeof payload === "object" && payload !== null && "detail" in payload
      ? (payload as { detail?: unknown }).detail
      : payload;

  if (isMappingPrompt(detail)) {
    throw new MappingRequiredError(detail);
  }

  if (typeof detail === "string") {
    throw new Error(detail);
  }

  throw new Error(`Request failed with status ${response.status}`);
};

export const uploadDataset = async (params: {
  file: File;
  dateColumn?: string;
  targetColumn?: string;
  itemColumn?: string;
}): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append("file", params.file);

  if (params.dateColumn) {
    formData.append("date_column", params.dateColumn);
  }
  if (params.targetColumn) {
    formData.append("target_column", params.targetColumn);
  }
  if (params.itemColumn) {
    formData.append("item_column", params.itemColumn);
  }

  const response = await fetch(buildUrl("/api/upload"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    return readErrorResponse(response);
  }

  return response.json() as Promise<AnalysisResult>;
};

export const sendChatMessage = async (runId: string, message: string): Promise<ChatResponse> => {
  const response = await fetch(buildUrl("/api/chat"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      run_id: runId,
      message,
    }),
  });

  if (!response.ok) {
    return readErrorResponse(response);
  }

  return response.json() as Promise<ChatResponse>;
};
