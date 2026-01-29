# TAPP - Intelligent Transcription & Analysis Platform

![Project Banner](assets/home_hero.png)

**TAPP** is a robust AI transcription studio that transforms raw audio into actionable insights. Built with **Flutter** (Web) and **FastAPI**, it demonstrates a modern full-stack architecture combining real-time UI interactivity with powerful server-side AI processing.

This project solves the challenge of unstructured audio data by providing **High-Fidelity Transcription**, **Speaker Diarization**, and **Generative AI Summaries** in a unified, professional interface.

---

## � Application Showcase

### **The Studio Experience**
The core of TAPP is the "Studio" - a glassmorphic workspace designed for focus.

| **Upload & Analyze** | **Multi-Language Support** |
|:---:|:---:|
| ![Transcribe View](assets/studio_main.png) | ![Language Selection](assets/language_opt.png) |
| *Drag-and-drop zone with animated hover states.* | *Support for 99+ languages including Hinglish.* |

### **Interactive Results**
Unlike static text files, TAPP's transcripts are fully interactive.

| **Diarized Transcript** | **AI Executive Summary** |
|:---:|:---:|
| ![Transcript View](assets/transcription.png) | ![AI Summary](assets/ai_summary.png) |
| *Click-to-seek audio synchronization.* | *Instant summarization via Gemini 1.5 Flash.* |

---

## 🏗️ Engineering & Architecture Decisions

This project was built to demonstrate scalable full-stack application development principles. Here are the key technical decisions made during development:

### **1. Hybrid "Brain-Body" Architecture**
*   **The Problem**: Browser-based transcription is slow and lacks accuracy for non-English languages.
*   **The Solution**: I decoupled the application into two distinct parts:
    *   **The Body (Frontend)**: Built with **Flutter Web** for a highly responsive, native-app-like feel. It handles audio playback, waveform visualization, and state management.
    *   **The Brain (Backend)**: Built with **FastAPI (Python)** to leverage the rich AI ecosystem. It runs **Faster-Whisper** (quantized for 4x speed) and integrates with **Google Gemini** for reasoning.

### **2. State Management with GetX**
*   **Decision**: Utilized **GetX** for state management instead of Provider or Bloc.
*   **Reasoning**: TAPP requires real-time synchronization between the Audio Player current timestamp and the highlighted transcript segment. GetX's reactive `Rx` variables allow for high-frequency UI updates (scroll-to-text) without significant performance overhead or boilerplate code.

### **3. Optimized AI Pipeline**
*   **Performance**: integrated `faster-whisper` with INT8 quantization. This reduced transcription time by ~60% compared to standard Whisper implementations while maintaining near-state-of-the-art accuracy.
*   **Cost-Efficiency**: Used **Gemini 1.5 Flash** for summarization, balancing extremely low latency with high reasoning capabilities for "Executive Summaries".

### **4. Modern UI/UX Design System**
*   **Aesthetic**: Implemented a **Glassmorphism** design language with deep gradients and blur filters (`BackdropFilter`).
*   **Accessibility**: Designed with a "Dark Mode First" approach to reduce eye strain for professionals working with text for long hours, featuring high-contrast text and clear active states.

---

## �️ Tech Stack Overview

### **Frontend**
*   **Framework**: Flutter (Web)
*   **State Management**: GetX
*   **Audio**: `audioplayers`
*   **Network**: HTTP (Multipart Requests)

### **Backend**
*   **Framework**: FastAPI (Python)
*   **ASR Model**: `faster-whisper` (OpenAI Whisper)
*   **LLM**: Google Generative AI (Gemini)
*   **Server**: Uvicorn / AIOHTTP

---

**Developed by Ali**
*Showcasing Agentic AI & Full-Stack Development Skills.*
