import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { MongoClient, ObjectId } from 'mongodb';
import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import * as pdfParse from 'pdf-parse';
import mammoth from 'mammoth';
import { createWorker } from 'tesseract.js';
import { v2 as cloudinary } from 'cloudinary';
import {
  buildImagePrompt,
  buildVideoPrompt,
  extractVisualOverrideDirectives,
} from './prompt-builders-optimized.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, '../.env') });

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

const app = express();
const port = Number(process.env.PORT || 4000);
const mongoUri = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017';
const dbName = process.env.MONGODB_DB_NAME || 'creative_studio_os';
const jwtSecret = process.env.JWT_SECRET || 'creative-studio-dev-secret';
const azureImageApiKey = process.env.AZURE_OPENAI_IMAGE_API_KEY || '';
const azureImageEndpoint = process.env.AZURE_OPENAI_IMAGE_ENDPOINT || '';
const azureImageDeployment = process.env.AZURE_OPENAI_IMAGE_DEPLOYMENT || 'gpt-image-2';
const azureImageApiVersion = process.env.AZURE_OPENAI_IMAGE_API_VERSION || '2024-02-01';
const azureVideoApiKey = process.env.AZURE_OPENAI_VIDEO_API_KEY || '';
const azureVideoEndpoint = process.env.AZURE_OPENAI_VIDEO_ENDPOINT || '';
const azureVideoModel = process.env.AZURE_OPENAI_VIDEO_MODEL || 'sora-2';
const azureVideoPollIntervalMs = Number(process.env.AZURE_OPENAI_VIDEO_POLL_INTERVAL_MS || 5000);
const azureVideoPollTimeoutMs = Number(process.env.AZURE_OPENAI_VIDEO_POLL_TIMEOUT_MS || 180000);
const azureVideoDownloadVariant = process.env.AZURE_OPENAI_VIDEO_DOWNLOAD_VARIANT || 'video';
const azureVideoDurationSeconds = String(process.env.AZURE_OPENAI_VIDEO_DURATION_SECONDS || '12').trim() || '12';

const uploadsDir = path.join(__dirname, 'uploads');
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });

app.use(cors());
app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));
app.use('/uploads', express.static(uploadsDir));

const createToken = (payload) => jwt.sign(payload, jwtSecret, { expiresIn: '7d' });
const nowIso = () => new Date().toISOString();
const normalizeEmail = (value) => String(value || '').trim().toLowerCase();
const superAdminEmail = normalizeEmail(process.env.SUPERADMIN_EMAIL);

const sanitizeUser = (user) => ({
  id: user.id || user._id?.toString?.() || user._id,
  email: user.email,
  full_name: user.full_name || '',
  company: user.company || '',
  role: user.role || 'user',
  status: user.status || 'active',
  plan_id: user.plan_id || null,
  plan_name: user.plan_name || null,
  created_at: user.created_at,
});

const sanitizeCompanyPersona = (persona) => ({
  id: persona.id || persona._id?.toString?.() || persona._id,
  user_id: persona.user_id,
  company: persona.company || '',
  name: persona.name || '',
  tagline: persona.tagline || '',
  logo_url: persona.logo_url || '',
  logo_placement: persona.logo_placement || 'none',
  preserve_original_logo: persona.preserve_original_logo !== false,
  audience: persona.audience || '',
  voice: persona.voice || '',
  goals: persona.goals || '',
  notes: persona.notes || '',
  visual_style_instructions: persona.visual_style_instructions || '',
  tuning_prompt: persona.tuning_prompt || '',
  learning_summary: persona.learning_summary || '',
  learning_count: Number(persona.learning_count || 0),
  analysis: persona.analysis || '',
  created_at: persona.created_at,
  updated_at: persona.updated_at,
});

const getPersonaLimitForPlan = (planName) => {
  switch (String(planName || 'free').trim().toLowerCase()) {
    case 'enterprise':
      return 20;
    case 'pro':
      return 5;
    default:
      return 1;
  }
};

const buildPersonaAnalysis = ({ company, tagline, audience, voice, goals, notes, visual_style_instructions, tuning_prompt, learning_summary }) => {
  return [
    `Brand identity: ${company || 'the company'}.`,
    tagline ? `Brand tagline: ${tagline}.` : null,
    audience ? `Primary audience: ${audience}.` : null,
    voice ? `Voice and tone: ${voice}.` : null,
    goals ? `Content goals: ${goals}.` : null,
    notes ? `Additional guidance: ${notes}.` : null,
    visual_style_instructions ? `Visual style instructions: ${visual_style_instructions}.` : null,
    tuning_prompt ? `Cross-platform brand style instructions: ${tuning_prompt}.` : null,
    learning_summary ? `Cross-platform learned writing preferences from prior generations: ${learning_summary}.` : null,
  ].filter(Boolean).join(' ');
};

const average = (values) => {
  const items = Array.isArray(values) ? values.filter((value) => Number.isFinite(value)) : [];
  if (items.length === 0) {
    return 0;
  }

  return items.reduce((total, value) => total + value, 0) / items.length;
};

const countMatches = (value, pattern) => {
  const matches = String(value || '').match(pattern);
  return matches ? matches.length : 0;
};

const summarizeGeneratedVariants = (variants) => {
  const items = Array.isArray(variants) ? variants : [];
  if (items.length === 0) {
    return '';
  }

  const contents = items
    .map((variant) => String(variant?.content || '').replace(/\s+/g, ' ').trim())
    .filter(Boolean);

  if (contents.length === 0) {
    return '';
  }

  const sentenceGroups = contents.map((content) => content.split(/(?<=[.!?])\s+/).filter(Boolean));
  const sentenceLengths = sentenceGroups.flat().map((sentence) => sentence.trim().split(/\s+/).filter(Boolean).length);
  const firstSentences = sentenceGroups.map((sentences) => sentences[0] || '');
  const paragraphCounts = contents.map((content) => content.split(/\n\s*\n/).filter(Boolean).length || 1);
  const lineCounts = contents.map((content) => content.split(/\n/).filter(Boolean).length || 1);
  const emojiTotal = contents.reduce((count, content) => count + countMatches(content, /[\u{1F300}-\u{1FAFF}]/gu), 0);
  const hashtagTotal = contents.reduce((count, content) => count + countMatches(content, /(^|\s)#[\p{L}\p{N}_]+/gu), 0);
  const exclamationTotal = contents.reduce((count, content) => count + countMatches(content, /!/g), 0);
  const questionHookCount = firstSentences.filter((sentence) => sentence.includes('?')).length;
  const ctaCount = contents.filter((content) => /\b(apply|join|learn more|discover|explore|contact|dm|message us|get started|sign up|book a demo|try it|read more|follow)\b/i.test(content)).length;
  const listStyleCount = contents.filter((content) => /[:\-]\s+[A-Z0-9]/.test(content) || /\n\s*[-*•]/.test(content)).length;
  const shortSentenceRatio = sentenceLengths.length ? sentenceLengths.filter((length) => length <= 10).length / sentenceLengths.length : 0;
  const longSentenceRatio = sentenceLengths.length ? sentenceLengths.filter((length) => length >= 20).length / sentenceLengths.length : 0;
  const averageWordCount = Math.round(items.reduce((total, variant) => total + Number(variant?.word_count || 0), 0) / items.length);
  const averageSentenceLength = Math.round(average(sentenceLengths));
  const averageParagraphCount = average(paragraphCounts);
  const averageLineCount = average(lineCounts);

  const styleSignals = [
    averageWordCount ? `Cross-platform preferred length: about ${averageWordCount} words before platform-specific adaptation.` : null,
    averageSentenceLength && averageSentenceLength <= 11 ? 'Sentence rhythm: prefers crisp, compact sentences.' : null,
    averageSentenceLength >= 18 ? 'Sentence rhythm: allows longer, more developed sentences when needed.' : null,
    shortSentenceRatio >= 0.6 ? 'Readability preference: keeps copy punchy and easy to scan.' : null,
    longSentenceRatio >= 0.35 ? 'Readability preference: mixes in fuller explanatory sentences for depth.' : null,
    questionHookCount >= Math.ceil(items.length / 2) ? 'Opening preference: often starts with a curiosity-driven or question-based hook.' : null,
    ctaCount >= Math.ceil(items.length / 2) ? 'Closing preference: usually ends with a direct call to action.' : null,
    listStyleCount >= Math.ceil(items.length / 2) ? 'Structure preference: often breaks ideas into list-like or segmented sections.' : null,
    averageParagraphCount >= 2 ? 'Layout preference: favors multi-paragraph flow over one dense block.' : null,
    averageLineCount >= 4 ? 'Formatting preference: uses visible line breaks to improve readability.' : null,
    emojiTotal > 0 ? 'Tone marker: may use light emoji emphasis when it suits the platform and audience.' : null,
    hashtagTotal >= items.length ? 'Packaging preference: often leaves room for hashtags when the platform supports them.' : null,
    exclamationTotal >= items.length ? 'Punctuation style: uses energetic emphasis sparingly through exclamation marks.' : null,
    'Memory rule: preserve reusable brand voice and writing behavior, but avoid storing temporary campaign facts, names, offers, locations, or topic-specific details.',
  ].filter(Boolean);

  return styleSignals.join(' ');
};

const mergeLearningSummary = (existingSummary, nextSummary) => {
  const current = String(existingSummary || '').trim();
  const incoming = String(nextSummary || '').trim();

  if (!incoming) {
    return current;
  }

  if (!current) {
    return incoming;
  }

  return `${incoming} ${current}`.slice(0, 1600).trim();
};

const trimHistoryVariantForList = (variant) => {
  if (!variant || typeof variant !== 'object') {
    return variant;
  }

  return {
    ...variant,
    content: typeof variant.content === 'string' ? variant.content.slice(0, 1200) : variant.content,
    image_base64: null,
  };
};

const trimHistoryMessageForList = (message) => {
  if (!message || typeof message !== 'object') {
    return message;
  }

  return {
    ...message,
    content: typeof message.content === 'string' ? message.content.slice(0, 600) : message.content,
    image_base64: null,
  };
};

const serializeHistoryListRow = (row) => ({
  ...row,
  id: row.id || row._id?.toString?.(),
  rag_context: typeof row.rag_context === 'string' ? row.rag_context.slice(0, 4000) : '',
  original_prompt: typeof row.original_prompt === 'string' ? row.original_prompt.slice(0, 2000) : '',
  variants: Array.isArray(row.variants) ? row.variants.map(trimHistoryVariantForList) : [],
  refinement_messages: Array.isArray(row.refinement_messages)
    ? row.refinement_messages.slice(-8).map(trimHistoryMessageForList)
    : [],
});

const buildHistoryConversationKey = (entry = {}) => {
  const explicitKey = String(entry.conversation_key || '').trim();
  if (explicitKey) {
    return explicitKey;
  }

  const rootId = String(entry.session_root_history_id || '').trim();
  if (rootId) {
    return rootId;
  }

  const topic = String(entry.topic || '').trim().toLowerCase();
  const persona = String(entry.persona || entry.persona_label || '').trim().toLowerCase();
  const contentType = String(entry.content_type || '').trim().toLowerCase();

  if (!topic && !persona && !contentType) {
    return '';
  }

  return [topic, persona, contentType].filter(Boolean).join('::');
};

const normalizeHistoryEntry = (entry = {}, userId) => {
  const createdDate = entry.created_date || nowIso();
  const updatedDate = nowIso();
  const conversationKey = buildHistoryConversationKey(entry);
  const sessionRootHistoryId = String(entry.session_root_history_id || '').trim();
  const refinementMessages = Array.isArray(entry.refinement_messages) ? entry.refinement_messages : [];
  const variants = Array.isArray(entry.variants) ? entry.variants : [];

  return {
    ...entry,
    user_id: userId,
    created_date: createdDate,
    updated_date: updatedDate,
    conversation_key: conversationKey || null,
    session_root_history_id: sessionRootHistoryId || null,
    refinement_messages: refinementMessages,
    refinement_message_count: refinementMessages.length,
    variants,
    variant_count: variants.length,
    latest_variant: variants[0] || null,
  };
};

const stripPersonaPaletteDirectives = (value) => {
  const text = String(value || '').trim();
  if (!text) {
    return '';
  }

  return text
    .replace(/\b(?:red|orange|blue|green|yellow|purple|pink|teal|cyan|magenta|gold|black|white|gray|grey|brown|beige)(?:\s*[-/]?\s*(?:led|based|dominant|primary|accent))?\s+palette\b/gi, '')
    .replace(/\b(?:red|orange|blue|green|yellow|purple|pink|teal|cyan|magenta|gold|black|white|gray|grey|brown|beige)(?:\s+theme(?:d)?)\b/gi, '')
    .replace(/\bpalette\s*:\s*[^.;]+/gi, '')
    .replace(/\bcolors?\s*:\s*[^.;]+/gi, '')
    .replace(/\b(?:use|prefer|keep|maintain|follow)\s+(?:a\s+)?(?:red|orange|blue|green|yellow|purple|pink|teal|cyan|magenta|gold|black|white|gray|grey|brown|beige)[^.;]*/gi, '')
    .replace(/\s{2,}/g, ' ')
    .replace(/\s+([,.;:])/g, '$1')
    .trim();
};

const normalizeAzureVideoEndpoint = (value) => String(value || '').replace(/\/+$/, '');

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

const extractAzureVideoUrl = (payload) => {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  return payload.video_url
    || payload.url
    || payload.output?.video_url
    || payload.output?.url
    || payload.output?.download_url
    || payload.output?.downloadUrl
    || payload.result?.video_url
    || payload.result?.url
    || payload.result?.download_url
    || payload.result?.downloadUrl
    || payload.data?.video_url
    || payload.data?.url
    || payload.data?.download_url
    || payload.data?.downloadUrl
    || payload.response?.video_url
    || payload.response?.url
    || payload.artifact?.url
    || payload.source?.url
    || payload.file?.url
    || payload.content?.url
    || (Array.isArray(payload.generations) ? payload.generations.find((item) => item?.url || item?.video_url || item?.download_url)?.url : null)
    || (Array.isArray(payload.generations) ? payload.generations.find((item) => item?.url || item?.video_url || item?.download_url)?.video_url : null)
    || (Array.isArray(payload.generations) ? payload.generations.find((item) => item?.url || item?.video_url || item?.download_url)?.download_url : null)
    || (Array.isArray(payload.output) ? payload.output.find((item) => item?.url || item?.video_url || item?.download_url)?.url : null)
    || (Array.isArray(payload.output) ? payload.output.find((item) => item?.url || item?.video_url || item?.download_url)?.video_url : null)
    || (Array.isArray(payload.output) ? payload.output.find((item) => item?.url || item?.video_url || item?.download_url)?.download_url : null)
    || (Array.isArray(payload.data) ? payload.data.find((item) => item?.url || item?.video_url || item?.download_url)?.url : null)
    || (Array.isArray(payload.data) ? payload.data.find((item) => item?.url || item?.video_url || item?.download_url)?.video_url : null)
    || (Array.isArray(payload.data) ? payload.data.find((item) => item?.url || item?.video_url || item?.download_url)?.download_url : null)
    || null;
};

const extractAzureVideoId = (payload) => {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  return payload.id || payload.video_id || payload.operation_id || payload.job_id || payload.data?.id || null;
};

const normalizeAzureVideoStatus = (payload) => {
  const rawStatus = String(
    payload?.status
      || payload?.state
      || payload?.job_status
      || payload?.data?.status
      || payload?.data?.state
      || ''
  ).trim().toLowerCase();

  if (!rawStatus) {
    return extractAzureVideoUrl(payload) ? 'completed' : 'submitted';
  }

  if (['succeeded', 'success', 'completed', 'complete', 'done'].includes(rawStatus)) {
    return 'completed';
  }

  if (['failed', 'error', 'cancelled', 'canceled', 'rejected'].includes(rawStatus)) {
    return 'failed';
  }

  if (['running', 'processing', 'queued', 'pending', 'submitted', 'in_progress', 'notstarted'].includes(rawStatus)) {
    return 'processing';
  }

  return rawStatus;
};

const buildAzureVideoStatusUrl = (endpoint, payload) => {
  const operationLocation = payload?.operation_location || payload?.operationLocation || payload?.headers?.operation_location || payload?.headers?.operationLocation;
  if (operationLocation) {
    return operationLocation;
  }

  const videoId = extractAzureVideoId(payload);
  if (!videoId) {
    return null;
  }

  return `${normalizeAzureVideoEndpoint(endpoint)}/${encodeURIComponent(videoId)}`;
};

const fetchAzureVideoStatus = async ({ statusUrl }) => {
  const response = await fetch(statusUrl, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      'api-key': azureVideoApiKey,
    },
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error?.message || data?.message || `Azure video status error: ${response.status} ${response.statusText}`);
  }

  return data;
};

const downloadAzureVideoContent = async ({ videoId, variant = azureVideoDownloadVariant }) => {
  const downloadUrl = `${normalizeAzureVideoEndpoint(azureVideoEndpoint)}/${encodeURIComponent(videoId)}/content?variant=${encodeURIComponent(variant)}`;
  const response = await fetch(downloadUrl, {
    method: 'GET',
    headers: {
      'api-key': azureVideoApiKey,
    },
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(errorText || `Azure video download error: ${response.status} ${response.statusText}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
};

const saveAzureVideoAsset = async ({ videoId, variant = azureVideoDownloadVariant }) => {
  const buffer = await downloadAzureVideoContent({ videoId, variant });

  const uploadResult = await new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      {
        folder: `${process.env.CLOUDINARY_FOLDER || 'creative-studio-os'}/videos`,
        resource_type: 'video',
        public_id: `${videoId}-${variant}`,
        overwrite: true,
      },
      (error, result) => {
        if (error) reject(error);
        else resolve(result);
      }
    );

    stream.end(buffer);
  });

  return uploadResult.secure_url;
};

const normalizeAzureVideoResult = async ({ payload, statusUrl = null }) => {
  const videoUrl = extractAzureVideoUrl(payload);
  const normalizedStatus = normalizeAzureVideoStatus(payload);
  const videoId = extractAzureVideoId(payload);
  let resolvedVideoUrl = videoUrl;

  if (!resolvedVideoUrl && normalizedStatus === 'completed' && videoId) {
    try {
      resolvedVideoUrl = await saveAzureVideoAsset({ videoId });
    } catch (error) {
      console.warn('Azure video download pending or unavailable:', error.message || error);
    }
  }

  return {
    video_url: resolvedVideoUrl,
    video_id: videoId,
    status: normalizedStatus === 'completed' && !resolvedVideoUrl ? 'processing' : normalizedStatus,
    status_url: statusUrl,
    provider_response: payload,
  };
};

const generateVideoWithAzure = async ({ prompt, durationSeconds = azureVideoDurationSeconds }) => {
  if (!azureVideoApiKey || !azureVideoEndpoint) {
    throw new Error('Azure video generation is not configured. Set AZURE_OPENAI_VIDEO_API_KEY and AZURE_OPENAI_VIDEO_ENDPOINT in the server environment.');
  }

  const normalizedDurationSeconds = ['4', '8', '12'].includes(String(durationSeconds || '').trim())
    ? String(durationSeconds).trim()
    : '12';

  const requestVideo = async (body) => {
    const response = await fetch(normalizeAzureVideoEndpoint(azureVideoEndpoint), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'api-key': azureVideoApiKey,
      },
      body: JSON.stringify(body),
    });

    const data = await response.json().catch(() => ({}));
    return { response, data };
  };

  let requestBody = {
    model: azureVideoModel,
    prompt,
    duration: normalizedDurationSeconds,
  };

  let { response, data } = await requestVideo(requestBody);

  if (!response.ok && /duration/i.test(String(data?.error?.message || data?.message || ''))) {
    requestBody = {
      model: azureVideoModel,
      prompt,
      seconds: normalizedDurationSeconds,
    };
    ({ response, data } = await requestVideo(requestBody));
  }

  if (!response.ok) {
    throw new Error(data?.error?.message || data?.message || `Azure video API error: ${response.status} ${response.statusText}`);
  }

  const initialStatusUrl = response.headers.get('operation-location') || response.headers.get('Operation-Location') || buildAzureVideoStatusUrl(azureVideoEndpoint, data);
  let normalized = await normalizeAzureVideoResult({ payload: data, statusUrl: initialStatusUrl });

  if (normalized.video_url || normalized.status === 'failed' || !normalized.status_url) {
    return normalized;
  }

  const startedAt = Date.now();
  while (Date.now() - startedAt < azureVideoPollTimeoutMs) {
    await sleep(azureVideoPollIntervalMs);
    const statusPayload = await fetchAzureVideoStatus({ statusUrl: normalized.status_url });
    normalized = await normalizeAzureVideoResult({ payload: statusPayload, statusUrl: normalized.status_url });

    if (normalized.video_url || normalized.status === 'completed' || normalized.status === 'failed') {
      return normalized;
    }
  }

  return {
    ...normalized,
    status: normalized.video_url ? 'completed' : 'processing',
  };
};

const getAzureVideoStatusById = async ({ videoId }) => {
  if (!azureVideoApiKey || !azureVideoEndpoint) {
    throw new Error('Azure video generation is not configured. Set AZURE_OPENAI_VIDEO_API_KEY and AZURE_OPENAI_VIDEO_ENDPOINT in the server environment.');
  }

  if (!String(videoId || '').trim()) {
    throw new Error('Video id is required');
  }

  const statusUrl = `${normalizeAzureVideoEndpoint(azureVideoEndpoint)}/${encodeURIComponent(String(videoId).trim())}`;
  const payload = await fetchAzureVideoStatus({ statusUrl });
  return await normalizeAzureVideoResult({ payload, statusUrl });
};

const ensureUploadsDir = async () => {
  await fs.mkdir(uploadsDir, { recursive: true });
};

let ocrWorkerPromise = null;
const ocrResultCache = new Map();
const activeOcrJobs = new Map();
const OCR_CACHE_LIMIT = 100;
const OCR_LANGUAGE_ALLOWLIST = new Set(['eng', 'spa', 'fra', 'deu', 'ita', 'por', 'nld']);
const OCR_AUTO_LANGUAGE = 'eng+spa+fra+deu+ita+por+nld';
const parsePdf = pdfParse.default || pdfParse;

const setOcrCache = (key, value) => {
  if (!key) {
    return;
  }

  if (ocrResultCache.has(key)) {
    ocrResultCache.delete(key);
  }

  ocrResultCache.set(key, value);

  if (ocrResultCache.size > OCR_CACHE_LIMIT) {
    const oldestKey = ocrResultCache.keys().next().value;
    if (oldestKey) {
      ocrResultCache.delete(oldestKey);
    }
  }
};

const getOcrCacheKey = ({ buffer, languages }) => `${languages}::${crypto.createHash('sha1').update(buffer).digest('hex')}`;

const normalizeOcrLanguages = (value) => {
  if (!String(value || '').trim()) {
    return OCR_AUTO_LANGUAGE;
  }

  const requested = String(value || 'eng')
    .split(/[+,\s]+/)
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean)
    .filter((item) => OCR_LANGUAGE_ALLOWLIST.has(item));

  return requested.length > 0 ? Array.from(new Set(requested)).join('+') : OCR_AUTO_LANGUAGE;
};

const getOcrWorker = async () => {
  if (!ocrWorkerPromise) {
    ocrWorkerPromise = (async () => {
      return new Map();
    })();
  }

  return ocrWorkerPromise;
};

const getOcrWorkerForLanguage = async (languages) => {
  const workers = await getOcrWorker();
  if (!workers.has(languages)) {
    workers.set(languages, (async () => {
      const worker = await createWorker(languages);
      return worker;
    })());
  }

  return workers.get(languages);
};

const parseDataUrlImage = (value) => {
  const match = String(value || '').match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
  if (!match) {
    return null;
  }

  return {
    mimeType: match[1],
    buffer: Buffer.from(match[2], 'base64'),
  };
};

const saveLogoUpload = async (logoFile) => {
  const parsed = parseDataUrlImage(logoFile);
  if (!parsed) {
    throw new Error('Invalid logo file payload');
  }

  const uploadResult = await cloudinary.uploader.upload(logoFile, {
    folder: `${process.env.CLOUDINARY_FOLDER || 'creative-studio-os'}/logos`,
    resource_type: 'image',
  });

  return uploadResult.secure_url;
};

const parseDataUrlFile = (value) => {
  const match = String(value || '').match(/^data:([^;]+);base64,(.+)$/);
  if (!match) {
    return null;
  }

  return {
    mimeType: match[1],
    buffer: Buffer.from(match[2], 'base64'),
  };
};

const extractTextFromBuffer = async ({ buffer, mimeType, fileName, ocrLanguages = OCR_AUTO_LANGUAGE }) => {
  const normalizedMimeType = String(mimeType || '').toLowerCase();
  const normalizedFileName = String(fileName || '').toLowerCase();
  const isTextLike = normalizedMimeType.startsWith('text/')
    || normalizedMimeType.includes('json')
    || normalizedMimeType.includes('xml')
    || normalizedFileName.endsWith('.md')
    || normalizedFileName.endsWith('.txt')
    || normalizedFileName.endsWith('.csv')
    || normalizedFileName.endsWith('.json');

  const isPdf = normalizedMimeType === 'application/pdf' || normalizedFileName.endsWith('.pdf');
  const isDocx = normalizedMimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    || normalizedFileName.endsWith('.docx');
  const isImage = normalizedMimeType.startsWith('image/')
    || /\.(png|jpe?g|webp|bmp|gif|tiff?)$/.test(normalizedFileName);

  if (isTextLike) {
    return buffer.toString('utf8').replace(/\u0000/g, '').trim();
  }

  if (isPdf) {
    const parsed = await parsePdf(buffer);
    return String(parsed?.text || '').replace(/\u0000/g, '').trim();
  }

  if (isDocx) {
    const parsed = await mammoth.extractRawText({ buffer });
    return String(parsed?.value || '').replace(/\u0000/g, '').trim();
  }

  if (isImage) {
    const cacheKey = getOcrCacheKey({ buffer, languages: ocrLanguages });
    if (ocrResultCache.has(cacheKey)) {
      return ocrResultCache.get(cacheKey);
    }

    const worker = await getOcrWorkerForLanguage(ocrLanguages);
    const result = await worker.recognize(buffer);
    const text = String(result?.data?.text || '').replace(/\u0000/g, '').replace(/\s+/g, ' ').trim();
    setOcrCache(cacheKey, text);
    return text;
  }

  throw new Error('Unsupported file type. Upload txt, md, csv, json, pdf, docx, or image files.');
};

const extractTextFromUrl = async (url) => {
  const response = await fetch(url, {
    headers: {
      'User-Agent': 'CreativeStudioOS/1.0',
      Accept: 'text/html, text/plain, application/json;q=0.9, */*;q=0.1',
    },
  });

  if (!response.ok) {
    throw new Error(`Unable to fetch URL: ${response.status} ${response.statusText}`);
  }

  const html = await response.text();
  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const bodyText = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/\s+/g, ' ')
    .trim();

  return {
    title: titleMatch ? titleMatch[1].replace(/\s+/g, ' ').trim() : '',
    content: bodyText,
  };
};

const normalizeKnowledgeText = (value) => String(value || '').replace(/\s+/g, ' ').trim();

const tokenizeKnowledge = (value) => {
  const matches = normalizeKnowledgeText(value).toLowerCase().match(/[a-z0-9]{2,}/g);
  return matches ? Array.from(new Set(matches)) : [];
};

const buildKnowledgeChunks = (content) => {
  const normalized = String(content || '').replace(/\r/g, '').trim();
  if (!normalized) {
    return [];
  }

  const paragraphs = normalized
    .split(/\n\s*\n/)
    .map((item) => item.trim())
    .filter(Boolean);

  const sourceUnits = paragraphs.length > 0 ? paragraphs : [normalized];
  const chunks = [];

  sourceUnits.forEach((unit) => {
    const sentences = unit.split(/(?<=[.!?])\s+/).filter(Boolean);
    if (sentences.length <= 3) {
      chunks.push(unit);
      return;
    }

    for (let index = 0; index < sentences.length; index += 3) {
      chunks.push(sentences.slice(index, index + 3).join(' ').trim());
    }
  });

  return chunks
    .map((chunk, index) => ({
      id: `${index + 1}`,
      text: chunk,
      tokens: tokenizeKnowledge(chunk),
    }))
    .filter((chunk) => chunk.text.length > 0);
};

const scoreKnowledgeChunk = (chunk, queryTokens) => {
  if (!chunk || !Array.isArray(chunk.tokens) || queryTokens.length === 0) {
    return 0;
  }

  const tokenSet = new Set(chunk.tokens);
  return queryTokens.reduce((score, token) => score + (tokenSet.has(token) ? 1 : 0), 0);
};

const buildRagContext = ({ knowledgeItems, query, limit = 4 }) => {
  const queryTokens = tokenizeKnowledge(query);
  const scored = [];

  (Array.isArray(knowledgeItems) ? knowledgeItems : []).forEach((item) => {
    (item.chunks || []).forEach((chunk) => {
      const score = scoreKnowledgeChunk(chunk, queryTokens);
      if (score > 0) {
        scored.push({
          sourceId: item.id || item._id?.toString?.() || '',
          sourceTitle: item.title || 'Knowledge Source',
          sourceType: item.source_type || 'text',
          text: chunk.text,
          score,
        });
      }
    });
  });

  return scored
    .sort((left, right) => right.score - left.score)
    .slice(0, limit)
    .map((item, index) => `[Source ${index + 1}: ${item.sourceTitle} | ${item.sourceType}] ${item.text}`)
    .join('\n');
};

const selectAzureImageSize = ({ platform, contentType }) => {
  const normalizedPlatform = String(platform?.label || '').trim().toLowerCase();
  const normalizedContentType = String(contentType || '').trim().toLowerCase();
  const isImageOnly = normalizedContentType === 'image' || normalizedContentType === 'image-only';
  const isTextAndImage = normalizedContentType.includes('text') && normalizedContentType.includes('image');

  const platformSizePresets = {
    linkedin: isImageOnly ? '1024x1792' : '1024x1024',
    instagram: '1024x1792',
    facebook: isImageOnly ? '1792x1024' : '1024x1024',
    youtube: '1792x1024',
    github: isTextAndImage ? '1024x1024' : '1792x1024',
    'x / twitter': isTextAndImage ? '1024x1024' : '1792x1024',
    threads: '1024x1792',
  };

  return platformSizePresets[normalizedPlatform] || '1024x1024';
};

const imageGenerationJobs = new Map();
const imageGenerationDurations = [];
const IMAGE_JOB_HISTORY_LIMIT = 25;

const createImageJobId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

const getAverageImageGenerationDurationMs = () => {
  if (imageGenerationDurations.length === 0) {
    return 120000;
  }

  return Math.round(imageGenerationDurations.reduce((total, value) => total + value, 0) / imageGenerationDurations.length);
};

const pushImageGenerationDuration = (durationMs) => {
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return;
  }

  imageGenerationDurations.push(durationMs);
  if (imageGenerationDurations.length > IMAGE_JOB_HISTORY_LIMIT) {
    imageGenerationDurations.shift();
  }
};

const getImageJobStatusPayload = (job) => {
  if (!job) {
    return null;
  }

  const now = Date.now();
  const elapsedMs = Math.max(0, now - job.startedAt);
  const estimatedTotalMs = job.completedAt
    ? Math.max(job.completedAt - job.startedAt, 1000)
    : Math.max(job.estimatedTotalMs || getAverageImageGenerationDurationMs(), 1000);
  const derivedProgress = Math.min(95, Math.max(job.progress || 0, Math.round((elapsedMs / estimatedTotalMs) * 100)));
  const derivedPhase = job.status === 'processing' && elapsedMs >= 4000
    ? 'Waiting for Azure image model response'
    : job.phase;
  const progress = job.status === 'completed'
    ? 100
    : job.status === 'failed'
      ? Math.min(job.progress || 0, 99)
      : derivedProgress;
  const remainingMs = job.status === 'completed'
    ? 0
    : Math.max(0, estimatedTotalMs - elapsedMs);

  return {
    id: job.id,
    status: job.status,
    phase: derivedPhase,
    progress,
    elapsedMs,
    estimatedTotalMs,
    estimatedRemainingMs: remainingMs,
    startedAt: new Date(job.startedAt).toISOString(),
    completedAt: job.completedAt ? new Date(job.completedAt).toISOString() : null,
    error: job.error || null,
    result: job.status === 'completed' ? job.result : null,
  };
};

const updateImageJob = (jobId, updates) => {
  const existing = imageGenerationJobs.get(jobId);
  if (!existing) {
    return null;
  }

  const nextJob = {
    ...existing,
    ...updates,
  };
  imageGenerationJobs.set(jobId, nextJob);
  return nextJob;
};

const startImageGenerationJob = ({ prompt, size }) => {
  const jobId = createImageJobId();
  const startedAt = Date.now();
  const estimatedTotalMs = getAverageImageGenerationDurationMs();

  imageGenerationJobs.set(jobId, {
    id: jobId,
    status: 'queued',
    phase: 'Queued for image generation',
    progress: 5,
    prompt,
    size,
    startedAt,
    estimatedTotalMs,
    completedAt: null,
    error: null,
    result: null,
  });

  void (async () => {
    try {
      updateImageJob(jobId, {
        status: 'processing',
        phase: 'Submitting request to Azure image model',
        progress: 15,
      });

      const phaseTimer = setTimeout(() => {
        updateImageJob(jobId, {
          status: 'processing',
          phase: 'Waiting for Azure image model response',
          progress: 35,
        });
      }, 1500);

      const image = await generateImageWithAzure({ prompt, size });
      clearTimeout(phaseTimer);
      const completedAt = Date.now();
      pushImageGenerationDuration(completedAt - startedAt);

      updateImageJob(jobId, {
        status: 'completed',
        phase: 'Image generated',
        progress: 100,
        completedAt,
        result: image,
      });
    } catch (error) {
      updateImageJob(jobId, {
        status: 'failed',
        phase: 'Image generation failed',
        completedAt: Date.now(),
        error: error.message || 'Image generation failed',
      });
    }
  })();

  return jobId;
};

const generateImageWithAzure = async ({ prompt, size = '1024x1024' }) => {
  if (!azureImageApiKey || !azureImageEndpoint) {
    throw new Error('Azure image generation is not configured on the server.');
  }

  const baseEndpoint = azureImageEndpoint.replace(/\/$/, '');
  const requestUrl = `${baseEndpoint}/openai/deployments/${azureImageDeployment}/images/generations?api-version=${azureImageApiVersion}`;
  const requestTimeoutMs = 180000;

  const requestImage = async (body) => {
    const controller = new AbortController();
    const timeoutHandle = setTimeout(() => controller.abort(), requestTimeoutMs);

    try {
      const response = await fetch(requestUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-key': azureImageApiKey,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      const data = await response.json().catch(() => ({}));
      return { response, data };
    } catch (error) {
      if (error?.name === 'AbortError') {
        throw new Error(`Azure image generation timed out after ${requestTimeoutMs / 1000} seconds.`);
      }

      throw error;
    } finally {
      clearTimeout(timeoutHandle);
    }
  };

  let requestBody = {
    prompt,
    size,
    quality: 'high',
  };

  let { response, data } = await requestImage(requestBody);

  if (!response.ok && /quality/i.test(String(data?.error?.message || ''))) {
    requestBody = {
      prompt,
      size,
    };
    ({ response, data } = await requestImage(requestBody));
  }

  if (!response.ok && /size|resolution|dimensions/i.test(String(data?.error?.message || '')) && size !== '1024x1024') {
    requestBody = {
      prompt,
      size: '1024x1024',
      quality: 'high',
    };
    ({ response, data } = await requestImage(requestBody));

    if (!response.ok && /quality/i.test(String(data?.error?.message || ''))) {
      requestBody = {
        prompt,
        size: '1024x1024',
      };
      ({ response, data } = await requestImage(requestBody));
    }
  }

  if (!response.ok) {
    throw new Error(data.error?.message || `Azure image API error: ${response.status} ${response.statusText}`);
  }

  const image = Array.isArray(data.data) ? data.data[0] : null;
  const imageUrl = image?.url || null;
  const b64Json = image?.b64_json || null;

  if (!imageUrl && !b64Json) {
    throw new Error('Azure image API returned no image payload.');
  }

  let finalImageUrl = imageUrl;

if (!finalImageUrl && b64Json) {
  const uploadResult = await cloudinary.uploader.upload(
    `data:image/png;base64,${b64Json}`,
    {
      folder: `${process.env.CLOUDINARY_FOLDER || 'creative-studio-os'}/images`,
      resource_type: 'image',
    }
  );

  finalImageUrl = uploadResult.secure_url;
}

return {
  image_url: finalImageUrl,
  revised_prompt: data?.data?.[0]?.revised_prompt || prompt,
};
};

const createMongoStore = (db) => ({
  type: 'mongo',
  async init() {
    const users = db.collection('users');
    const history = db.collection('content_history');

    await Promise.all([
      history.createIndex({ user_id: 1, created_date: -1 }),
      history.createIndex({ created_date: -1 }),
      history.createIndex({ user_id: 1, updated_date: -1, created_date: -1 }),
      history.createIndex({ updated_date: -1, created_date: -1 }),
      history.createIndex({ user_id: 1, conversation_key: 1 }),
      history.createIndex({ user_id: 1, session_root_history_id: 1 }),
    ]);

    if (!superAdminEmail) {
      console.warn('SUPERADMIN_EMAIL is not set; superadmin login will remain unavailable until configured.');
      return;
    }

    const existingAdmin = await users.findOne({ email: superAdminEmail, role: 'superadmin' });
    if (!existingAdmin) {
      console.warn(`Super admin user ${superAdminEmail} was not found in MongoDB; superadmin login will remain unavailable until seeded.`);
    }
  },
  async getHealth() {
    return { ok: true, users: await db.collection('users').countDocuments(), mode: 'mongo' };
  },
  async findUserByEmail(email) {
    return await db.collection('users').findOne({ email: normalizeEmail(email) });
  },
  async findUserById(id) {
    return await db.collection('users').findOne({ _id: new ObjectId(id) });
  },
  async insertUser(user) {
    const result = await db.collection('users').insertOne(user);
    return await db.collection('users').findOne({ _id: result.insertedId });
  },
  async countUsers() {
    return await db.collection('users').countDocuments();
  },
  async listUsers(filter = {}) {
    const rows = await db.collection('users').find({}).toArray();
    return typeof filter === 'function' ? rows.filter(filter) : rows.filter((row) => Object.entries(filter).every(([key, value]) => row[key] === value));
  },
  async listPlans() {
    return await db.collection('plans').find({}).sort({ created_at: 1 }).toArray();
  },
  async findPlanByName(name) {
    return await db.collection('plans').findOne({ name });
  },
  async listHistory(filter = {}, sortField = 'created_date', limit, offset = 0) {
    const collection = db.collection('content_history');

    if (typeof filter === 'function') {
      const rows = await collection.find({}, { allowDiskUse: true }).sort({ [sortField]: -1 }).skip(offset).limit(limit || 500).toArray();
      return rows.filter(filter);
    }

    const cursor = collection.find(filter || {}, { allowDiskUse: true }).sort({ [sortField]: -1 });
    if (typeof offset === 'number' && Number.isFinite(offset) && offset > 0) {
      cursor.skip(offset);
    }
    if (typeof limit === 'number' && Number.isFinite(limit) && limit > 0) {
      cursor.limit(limit);
    }

    return await cursor.toArray();
  },
  async listHistoryPage(filter = {}, sortField = 'created_date', limit = 10, beforeValue) {
    const collection = db.collection('content_history');
    const query = typeof filter === 'function' ? {} : { ...(filter || {}) };

    if (beforeValue) {
      query[sortField] = { $lt: beforeValue };
    }

    const rows = await collection
      .find(query, { allowDiskUse: true })
      .sort({ [sortField]: -1 })
      .limit(limit)
      .toArray();

    if (typeof filter === 'function') {
      return rows.filter(filter);
    }

    return rows;
  },
  async upsertHistoryConversation(entry) {
    const collection = db.collection('content_history');
    const conversationKey = String(entry.conversation_key || '').trim();
    const sessionRootHistoryId = String(entry.session_root_history_id || '').trim();

    if (!conversationKey && !sessionRootHistoryId) {
      return await this.insertHistory(entry);
    }

    const query = { user_id: entry.user_id };
    if (sessionRootHistoryId) {
      query.$or = [
        { session_root_history_id: sessionRootHistoryId },
        { conversation_key: conversationKey || sessionRootHistoryId },
      ];
    } else {
      query.conversation_key = conversationKey;
    }

    const existing = await collection.findOne(query, { sort: { updated_date: -1, created_date: -1 } });
    if (!existing) {
      const inserted = await this.insertHistory(entry);
      if (!inserted.session_root_history_id) {
        await collection.updateOne(
          { _id: inserted._id },
          { $set: { session_root_history_id: inserted._id.toString(), conversation_key: conversationKey || inserted._id.toString() } }
        );
        return await collection.findOne({ _id: inserted._id });
      }
      return inserted;
    }

    const nextSessionRootHistoryId = existing.session_root_history_id || existing._id?.toString?.() || sessionRootHistoryId || null;
    const nextConversationKey = conversationKey || existing.conversation_key || nextSessionRootHistoryId;
    await collection.updateOne(
      { _id: existing._id },
      {
        $set: {
          ...entry,
          session_root_history_id: nextSessionRootHistoryId,
          conversation_key: nextConversationKey,
          created_date: existing.created_date || entry.created_date,
          updated_date: nowIso(),
        },
      }
    );

    return await collection.findOne({ _id: existing._id });
  },
  async insertHistory(entry) {
    const result = await db.collection('content_history').insertOne(entry);
    return await db.collection('content_history').findOne({ _id: result.insertedId });
  },
  async updateHistoryStatus(id, userId) {
    await db.collection('content_history').updateOne(
      { _id: new ObjectId(id), user_id: userId },
      { $set: { status: 'deleted', deleted_at: nowIso() } }
    );
    return await db.collection('content_history').findOne({ _id: new ObjectId(id) });
  },
  async listCompanyPersonas(userId) {
    return await db.collection('company_personas').find({ user_id: userId }).sort({ created_at: -1 }).toArray();
  },
  async countCompanyPersonas(userId) {
    return await db.collection('company_personas').countDocuments({ user_id: userId });
  },
  async findCompanyPersonaById(id, userId) {
    return await db.collection('company_personas').findOne({ _id: new ObjectId(id), user_id: userId });
  },
  async insertCompanyPersona(persona) {
    const result = await db.collection('company_personas').insertOne(persona);
    return await db.collection('company_personas').findOne({ _id: result.insertedId });
  },
  async updateCompanyPersona(id, userId, updates) {
    await db.collection('company_personas').updateOne(
      { _id: new ObjectId(id), user_id: userId },
      { $set: updates }
    );
    return await db.collection('company_personas').findOne({ _id: new ObjectId(id), user_id: userId });
  },
  async deleteCompanyPersona(id, userId) {
    const result = await db.collection('company_personas').deleteOne({ _id: new ObjectId(id), user_id: userId });
    return result.deletedCount > 0;
  },
  async listKnowledgeSources(userId) {
    return await db.collection('knowledge_sources').find({ user_id: userId }).sort({ updated_at: -1 }).toArray();
  },
  async findKnowledgeSourceById(id, userId) {
    return await db.collection('knowledge_sources').findOne({ _id: new ObjectId(id), user_id: userId });
  },
  async insertKnowledgeSource(source) {
    const result = await db.collection('knowledge_sources').insertOne(source);
    return await db.collection('knowledge_sources').findOne({ _id: result.insertedId });
  },
  async updateKnowledgeSource(id, userId, updates) {
    await db.collection('knowledge_sources').updateOne(
      { _id: new ObjectId(id), user_id: userId },
      { $set: updates }
    );
    return await db.collection('knowledge_sources').findOne({ _id: new ObjectId(id), user_id: userId });
  },
  async deleteKnowledgeSource(id, userId) {
    const result = await db.collection('knowledge_sources').deleteOne({ _id: new ObjectId(id), user_id: userId });
    return result.deletedCount > 0;
  },
});

let store = null;

const authRequired = async (req, res, next) => {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;

  if (!token) {
    return res.status(401).json({ message: 'Authentication required' });
  }

  try {
    const decoded = jwt.verify(token, jwtSecret);
    const user = await store.findUserById(decoded.sub);

    if (!user) {
      return res.status(401).json({ message: 'User not found' });
    }

    req.user = user;
    next();
  } catch (error) {
    return res.status(401).json({ message: 'Invalid session' });
  }
};

const superAdminRequired = async (req, res, next) => {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;

  if (!token) {
    return res.status(401).json({ message: 'Authentication required' });
  }

  try {
    const decoded = jwt.verify(token, jwtSecret);
    if (decoded.role !== 'superadmin') {
      return res.status(403).json({ message: 'Super admin access required' });
    }

    req.superAdmin = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ message: 'Invalid session' });
  }
};

app.get('/api/health', async (_req, res) => {
  const health = await store.getHealth();
  res.json(health);
});

app.post('/api/auth/register', async (req, res) => {
  const { email, password, fullName, company } = req.body;
  const normalizedEmail = normalizeEmail(email);

  if (!normalizedEmail || !password || !fullName || !company) {
    return res.status(400).json({ message: 'Missing required fields' });
  }

  const existing = await store.findUserByEmail(normalizedEmail);
  if (existing) {
    return res.status(409).json({ message: 'User already registered' });
  }

  const freePlan = await store.findPlanByName('Free');
  if (!freePlan) {
    return res.status(500).json({ message: 'Default Free plan is missing in MongoDB' });
  }

  const password_hash = await bcrypt.hash(password, 10);
  const created_at = nowIso();
  const insertedUser = await store.insertUser({
    email: normalizedEmail,
    full_name: fullName,
    company,
    role: 'user',
    status: 'active',
    password_hash,
    created_at,
    plan_id: freePlan.id || freePlan._id?.toString?.() || null,
    plan_name: freePlan.name,
  });

  const user = insertedUser?._id ? insertedUser : (await store.findUserByEmail(normalizedEmail));
  const token = createToken({ sub: user.id || user._id.toString(), role: user.role, email: user.email });
  res.status(201).json({ token, user: sanitizeUser(user) });
});

app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;
  const normalizedEmail = normalizeEmail(email);
  const user = await store.findUserByEmail(normalizedEmail);

  if (!user) {
    return res.status(401).json({ message: 'Invalid email or password' });
  }

  const isValid = await bcrypt.compare(password || '', user.password_hash || '');
  if (!isValid) {
    return res.status(401).json({ message: 'Invalid email or password' });
  }

  const token = createToken({ sub: user.id || user._id.toString(), role: user.role, email: user.email });
  res.json({ token, user: sanitizeUser(user) });
});

app.get('/api/auth/session', authRequired, async (req, res) => {
  res.json({ user: sanitizeUser(req.user) });
});

app.get('/api/user/metrics', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const rows = await store.listHistory(
    (row) => String(row.user_id) === String(userId),
    'created_date'
  );

  const now = new Date();
  const thisMonthStart = new Date(now.getFullYear(), now.getMonth(), 1);
  const generationsThisMonth = rows.filter(
    (row) => new Date(row.created_date) >= thisMonthStart
  ).length;
  const companyPersonaCount = await store.countCompanyPersonas(userId);
  const planName = req.user.plan_name || 'Free';

  res.json({
    generationsThisMonth,
    totalGenerations: rows.length,
    planName,
    planId: req.user.plan_id || null,
    companyPersonaCount,
    companyPersonaLimit: getPersonaLimitForPlan(planName),
  });
});

app.get('/api/company-personas', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const rows = await store.listCompanyPersonas(userId);
  res.json(rows.map(sanitizeCompanyPersona));
});

app.post('/api/company-personas/logo', authRequired, async (req, res) => {
  const { fileName, fileData } = req.body || {};

  if (!fileName || !fileData) {
    return res.status(400).json({ message: 'Logo file is required' });
  }

  try {
    const logoUrl = await saveLogoUpload(fileData);
    res.status(201).json({ logoUrl, fileName });
  } catch (error) {
    res.status(400).json({ message: error.message || 'Unable to upload logo' });
  }
});

app.post('/api/company-personas', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const { name, company, tagline, logo_url, logo_placement, preserve_original_logo, audience, voice, goals, notes, visual_style_instructions, tuning_prompt } = req.body || {};

  if (!name || !company) {
    return res.status(400).json({ message: 'Persona name and company are required' });
  }

  const existingCount = await store.countCompanyPersonas(userId);
  const planName = req.user.plan_name || 'Free';
  const personaLimit = getPersonaLimitForPlan(planName);
  if (existingCount >= personaLimit) {
    return res.status(403).json({ message: `Your ${planName} plan allows up to ${personaLimit} company persona${personaLimit > 1 ? 's' : ''}.` });
  }

  const timestamp = nowIso();
  const persona = await store.insertCompanyPersona({
    user_id: userId,
    company: String(company).trim(),
    name: String(name).trim(),
    tagline: String(tagline || '').trim(),
    logo_url: String(logo_url || '').trim(),
    logo_placement: String(logo_placement || 'none').trim() || 'none',
    preserve_original_logo: preserve_original_logo !== false,
    audience: String(audience || '').trim(),
    voice: String(voice || '').trim(),
    goals: String(goals || '').trim(),
    notes: String(notes || '').trim(),
    visual_style_instructions: String(visual_style_instructions || '').trim(),
    tuning_prompt: String(tuning_prompt || '').trim(),
    learning_summary: '',
    learning_count: 0,
    analysis: buildPersonaAnalysis({ name, company, tagline, audience, voice, goals, notes, visual_style_instructions, tuning_prompt }),
    created_at: timestamp,
    updated_at: timestamp,
  });

  res.status(201).json(sanitizeCompanyPersona(persona));
});

app.patch('/api/company-personas/:id', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const existing = await store.findCompanyPersonaById(req.params.id, userId);

  if (!existing) {
    return res.status(404).json({ message: 'Company persona not found' });
  }

  const nextValues = {
    name: String(req.body.name ?? existing.name).trim(),
    company: String(req.body.company ?? existing.company).trim(),
    tagline: String(req.body.tagline ?? existing.tagline).trim(),
    logo_url: String(req.body.logo_url ?? existing.logo_url).trim(),
    logo_placement: String(req.body.logo_placement ?? existing.logo_placement ?? 'none').trim() || 'none',
    preserve_original_logo: req.body.preserve_original_logo !== undefined ? req.body.preserve_original_logo !== false : existing.preserve_original_logo !== false,
    audience: String(req.body.audience ?? existing.audience).trim(),
    voice: String(req.body.voice ?? existing.voice).trim(),
    goals: String(req.body.goals ?? existing.goals).trim(),
    notes: String(req.body.notes ?? existing.notes).trim(),
    visual_style_instructions: String(req.body.visual_style_instructions ?? existing.visual_style_instructions).trim(),
    tuning_prompt: String(req.body.tuning_prompt ?? existing.tuning_prompt).trim(),
    learning_summary: String(req.body.learning_summary ?? existing.learning_summary).trim(),
    learning_count: Number(req.body.learning_count ?? existing.learning_count ?? 0),
  };

  const updated = await store.updateCompanyPersona(req.params.id, userId, {
    ...nextValues,
    analysis: buildPersonaAnalysis(nextValues),
    updated_at: nowIso(),
  });

  res.json(sanitizeCompanyPersona(updated));
});

app.delete('/api/company-personas/:id', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const deleted = await store.deleteCompanyPersona(req.params.id, userId);

  if (!deleted) {
    return res.status(404).json({ message: 'Company persona not found' });
  }

  res.json({ ok: true });
});

app.post('/api/company-personas/:id/learn', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const existing = await store.findCompanyPersonaById(req.params.id, userId);

  if (!existing) {
    return res.status(404).json({ message: 'Company persona not found' });
  }

  const generatedSummary = summarizeGeneratedVariants(req.body?.variants);
  const manualFeedback = String(req.body?.feedback || '').trim();
  const nextLearningSummary = mergeLearningSummary(
    existing.learning_summary,
    [manualFeedback, generatedSummary].filter(Boolean).join(' ')
  );

  const nextValues = {
    name: existing.name,
    company: existing.company,
    tagline: existing.tagline,
    logo_url: existing.logo_url,
    logo_placement: existing.logo_placement || 'none',
    preserve_original_logo: existing.preserve_original_logo !== false,
    audience: existing.audience,
    voice: existing.voice,
    goals: existing.goals,
    notes: existing.notes,
    visual_style_instructions: existing.visual_style_instructions || '',
    tuning_prompt: existing.tuning_prompt || '',
    learning_summary: nextLearningSummary,
    learning_count: Number(existing.learning_count || 0) + 1,
  };

  const updated = await store.updateCompanyPersona(req.params.id, userId, {
    ...nextValues,
    analysis: buildPersonaAnalysis(nextValues),
    updated_at: nowIso(),
  });

  res.json(sanitizeCompanyPersona(updated));
});

app.post('/api/generate-image', authRequired, async (req, res) => {
  try {
    const platform = req.body?.platform || null;
    const topic = String(req.body?.topic || '').trim();
    const contentType = String(req.body?.contentType || '').trim();
    const companyPersona = req.body?.companyPersona || null;
    const logoPlacement = String(req.body?.logoPlacement || '').trim();
    const useOriginalLogo = req.body?.useOriginalLogo !== false;
    const ragContext = String(req.body?.ragContext || '').trim();
    const keywords = String(req.body?.keywords || '').trim();
    const variantTitle = String(req.body?.variantTitle || '').trim();
    const variantContent = String(req.body?.variantContent || '').trim();

    if (!topic) {
      return res.status(400).json({ message: 'Topic is required for image generation' });
    }

    const prompt = buildImagePrompt({
      platform,
      topic,
      companyPersona: companyPersona ? {
        ...companyPersona,
        logoPlacementOverride: logoPlacement === 'persona-default' ? (companyPersona.logo_placement || 'none') : (logoPlacement || companyPersona.logo_placement || 'none'),
        useOriginalLogo,
      } : null,
      contentType,
      ragContext,
      keywords,
      variantTitle,
      variantContent,
    });

    const size = selectAzureImageSize({ platform, contentType });

    if (req.body?.async !== false) {
      const jobId = startImageGenerationJob({ prompt, size });
      return res.status(202).json({
        jobId,
        prompt,
        size,
        status: getImageJobStatusPayload(imageGenerationJobs.get(jobId)),
      });
    }

    const image = await generateImageWithAzure({ prompt, size });
    res.json({
      prompt,
      size,
      ...image,
    });
  } catch (error) {
    res.status(500).json({ message: error.message || 'Image generation failed' });
  }
});

app.get('/api/generate-image/:jobId/status', authRequired, async (req, res) => {
  const job = imageGenerationJobs.get(String(req.params.jobId || '').trim());
  if (!job) {
    return res.status(404).json({ message: 'Image generation job not found' });
  }

  return res.json(getImageJobStatusPayload(job));
});

app.post('/api/generate-video', authRequired, async (req, res) => {
  try {
    const platform = req.body?.platform || null;
    const topic = String(req.body?.topic || '').trim();
    const contentType = String(req.body?.contentType || '').trim();
    const companyPersona = req.body?.companyPersona || null;
    const logoPlacement = String(req.body?.logoPlacement || '').trim();
    const useOriginalLogo = req.body?.useOriginalLogo !== false;
    const ragContext = String(req.body?.ragContext || '').trim();
    const keywords = String(req.body?.keywords || '').trim();
    const variantTitle = String(req.body?.variantTitle || '').trim();
    const variantContent = String(req.body?.variantContent || '').trim();

    if (!topic) {
      return res.status(400).json({ message: 'Topic is required for video generation' });
    }

    const prompt = buildVideoPrompt({
      platform,
      topic,
      companyPersona: companyPersona ? {
        ...companyPersona,
        logoPlacementOverride: logoPlacement === 'persona-default' ? (companyPersona.logo_placement || 'none') : (logoPlacement || companyPersona.logo_placement || 'none'),
        useOriginalLogo,
      } : null,
      contentType,
      ragContext,
      keywords,
      variantTitle,
      variantContent,
    });

    const video = await generateVideoWithAzure({ prompt });
    res.json({ prompt, ...video });
  } catch (error) {
    console.error('Video generation failed:', error?.message || error);
    res.status(500).json({ message: error.message || 'Video generation failed' });
  }
});

app.get('/api/video-status/:id', authRequired, async (req, res) => {
  try {
    const result = await getAzureVideoStatusById({ videoId: req.params.id });
    res.json(result);
  } catch (error) {
    console.error('Video status error:', error);
    res.status(500).json({ error: error.message || 'Failed to fetch video status' });
  }
});

app.get('/api/knowledge-sources', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const rows = await store.listKnowledgeSources(userId);
  res.json(rows.map((row) => ({ ...row, id: row.id || row._id?.toString?.() })));
});

app.get('/api/knowledge-sources/:id', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const source = await store.findKnowledgeSourceById(req.params.id, userId);

  if (!source) {
    return res.status(404).json({ message: 'Knowledge source not found' });
  }

  res.json({ ...source, id: source.id || source._id?.toString?.() });
});

app.post('/api/knowledge-sources', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const title = String(req.body?.title || '').trim();
  const content = String(req.body?.content || '').trim();
  const sourceType = String(req.body?.source_type || 'text').trim() || 'text';
  const tags = Array.isArray(req.body?.tags)
    ? req.body.tags.map((tag) => String(tag || '').trim()).filter(Boolean)
    : [];

  if (!title || !content) {
    return res.status(400).json({ message: 'Title and content are required' });
  }

  const timestamp = nowIso();
  const source = await store.insertKnowledgeSource({
    user_id: userId,
    title,
    content,
    source_type: sourceType,
    tags,
    chunks: buildKnowledgeChunks(content),
    created_at: timestamp,
    updated_at: timestamp,
  });

  res.status(201).json({ ...source, id: source.id || source._id?.toString?.() });
});

app.post('/api/knowledge-sources/ingest-url', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const url = String(req.body?.url || '').trim();
  const sourceType = String(req.body?.source_type || 'text').trim() || 'text';
  const tags = Array.isArray(req.body?.tags)
    ? req.body.tags.map((tag) => String(tag || '').trim()).filter(Boolean)
    : [];

  if (!url) {
    return res.status(400).json({ message: 'URL is required' });
  }

  try {
    const extracted = await extractTextFromUrl(url);
    if (!extracted.content) {
      return res.status(400).json({ message: 'No readable content found at the URL' });
    }

    const timestamp = nowIso();
    const source = await store.insertKnowledgeSource({
      user_id: userId,
      title: String(req.body?.title || extracted.title || url).trim(),
      content: extracted.content,
      source_type: sourceType,
      tags,
      source_url: url,
      ingestion_method: 'url',
      chunks: buildKnowledgeChunks(extracted.content),
      created_at: timestamp,
      updated_at: timestamp,
    });

    res.status(201).json({ ...source, id: source.id || source._id?.toString?.() });
  } catch (error) {
    res.status(400).json({ message: error.message || 'Unable to ingest URL' });
  }
});

app.post('/api/knowledge-sources/ingest-file', authRequired, upload.single('file'), async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const fileName = String(req.body?.fileName || req.file?.originalname || '').trim();
  const sourceType = String(req.body?.source_type || 'text').trim() || 'text';
  const ocrLanguages = normalizeOcrLanguages(req.body?.ocr_languages);
  const tags = Array.isArray(req.body?.tags)
    ? req.body.tags.map((tag) => String(tag || '').trim()).filter(Boolean)
    : typeof req.body?.tags === 'string'
      ? req.body.tags.split(',').map((tag) => String(tag || '').trim()).filter(Boolean)
    : [];

  if (!fileName || !req.file) {
    return res.status(400).json({ message: 'File upload is required' });
  }

  try {
    const timestamp = nowIso();
    const isImage = req.file.mimetype?.startsWith('image/') || /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(fileName);
    const initialContent = isImage ? '' : await extractTextFromBuffer({
      buffer: req.file.buffer,
      mimeType: req.file.mimetype,
      fileName,
      ocrLanguages,
    });

    if (!isImage && !initialContent) {
      return res.status(400).json({ message: 'No readable content found in the uploaded file' });
    }

    const source = await store.insertKnowledgeSource({
      user_id: userId,
      title: String(req.body?.title || fileName).trim(),
      content: initialContent,
      source_type: sourceType,
      tags,
      source_file_name: fileName,
      ingestion_method: 'file',
      chunks: buildKnowledgeChunks(initialContent),
      ocr_status: isImage ? 'processing' : 'completed',
      ocr_languages: isImage ? ocrLanguages : null,
      created_at: timestamp,
      updated_at: timestamp,
    });

    const sourceId = source.id || source._id?.toString?.();

    if (isImage && sourceId) {
      const jobKey = `${userId}:${sourceId}`;
      if (!activeOcrJobs.has(jobKey)) {
        activeOcrJobs.set(jobKey, (async () => {
          try {
            const content = await extractTextFromBuffer({
              buffer: req.file.buffer,
              mimeType: req.file.mimetype,
              fileName,
              ocrLanguages,
            });

            await store.updateKnowledgeSource(sourceId, userId, {
              content,
              chunks: buildKnowledgeChunks(content),
              ocr_status: content ? 'completed' : 'empty',
              updated_at: nowIso(),
            });
          } catch (error) {
            await store.updateKnowledgeSource(sourceId, userId, {
              ocr_status: 'failed',
              ocr_error: error.message || 'OCR failed',
              updated_at: nowIso(),
            });
          } finally {
            activeOcrJobs.delete(jobKey);
          }
        })());
      }
    }

    res.status(201).json({ ...source, id: source.id || source._id?.toString?.() });
  } catch (error) {
    res.status(400).json({ message: error.message || 'Unable to ingest file' });
  }
});

app.patch('/api/knowledge-sources/:id', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const existing = await store.findKnowledgeSourceById(req.params.id, userId);

  if (!existing) {
    return res.status(404).json({ message: 'Knowledge source not found' });
  }

  const title = String(req.body?.title ?? existing.title).trim();
  const content = String(req.body?.content ?? existing.content).trim();
  const sourceType = String(req.body?.source_type ?? existing.source_type ?? 'text').trim() || 'text';
  const tags = Array.isArray(req.body?.tags)
    ? req.body.tags.map((tag) => String(tag || '').trim()).filter(Boolean)
    : Array.isArray(existing.tags)
    ? existing.tags
    : [];

  const updated = await store.updateKnowledgeSource(req.params.id, userId, {
    title,
    content,
    source_type: sourceType,
    tags,
    chunks: buildKnowledgeChunks(content),
    updated_at: nowIso(),
  });

  res.json({ ...updated, id: updated.id || updated._id?.toString?.() });
});

app.delete('/api/knowledge-sources/:id', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const deleted = await store.deleteKnowledgeSource(req.params.id, userId);

  if (!deleted) {
    return res.status(404).json({ message: 'Knowledge source not found' });
  }

  res.json({ ok: true });
});

app.post('/api/rag/context', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const query = String(req.body?.query || '').trim();

  if (!query) {
    return res.status(400).json({ message: 'Query is required' });
  }

  const knowledgeItems = await store.listKnowledgeSources(userId);
  const context = buildRagContext({ knowledgeItems, query, limit: 6 });
  res.json({ context, count: context ? context.split('\n').length : 0 });
});

app.get('/api/history', authRequired, async (req, res) => {
  const limit = Math.min(Number(req.query.limit || 10), 50);
  const before = String(req.query.before || '').trim();
  const userId = req.user.id || req.user._id.toString();
  const rows = await store.listHistoryPage({ user_id: userId }, 'created_date', limit, before || undefined);
  const serializedRows = rows.map(serializeHistoryListRow);
  const nextCursor = rows.length === limit ? rows[rows.length - 1]?.created_date || null : null;

  res.json({ items: serializedRows, nextCursor });
});

app.post('/api/history', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  const payload = normalizeHistoryEntry({
    ...req.body,
    user_id: userId,
    user_name: req.user.full_name || req.user.email,
    user_email: req.user.email,
    status: req.body.status || 'completed',
  }, userId);

  const row = await store.upsertHistoryConversation(payload);
  res.status(201).json({ ...row, id: row.id || row._id?.toString?.() });
});

app.patch('/api/history/:id/delete', authRequired, async (req, res) => {
  const userId = req.user.id || req.user._id.toString();
  await store.updateHistoryStatus(req.params.id, userId);
  res.json({ ok: true });
});

app.get('/api/superadmin/metrics', superAdminRequired, async (_req, res) => {
  const users = await store.listUsers((user) => user.role !== 'superadmin');
  const rows = await store.listHistory({}, 'created_date', 200);
  res.json({ users, rows });
});

app.get('/api/superadmin/plans', superAdminRequired, async (_req, res) => {
  const plans = await store.listPlans();
  res.json(plans.map((plan) => ({ ...plan, id: plan.id || plan._id?.toString?.() })));
});

const tryStartMongo = async () => {
  const client = new MongoClient(mongoUri, {
    serverSelectionTimeoutMS: 5000,
  });

  await client.connect();
  const db = client.db(dbName);
  store = createMongoStore(db);
  await store.init();
  return client;
};

const start = async () => {
  await tryStartMongo();
  const server = app.listen(port, () => {
    console.log(`Mongo API listening on http://localhost:${port}`);
    console.log(`Using MongoDB at ${mongoUri}`);
    console.log(`\n✓ MongoDB connected successfully!`);
    console.log(`✓ Open MongoDB Compass and connect to: ${mongoUri}`);
    console.log(`✓ Or visit http://localhost:${port} in your browser\n`);
  });

  server.on('error', (error) => {
    if (error?.code === 'EADDRINUSE') {
      console.error(`Port ${port} is already in use. Set PORT to a free port before starting the server.`);
      process.exit(1);
    }

    console.error('Failed to start HTTP server', error);
    process.exit(1);
  });
};

start().catch((error) => {
  console.error('Failed to start server', error);
  process.exit(1);
});
