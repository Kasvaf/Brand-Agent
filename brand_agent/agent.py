"""
Brand Agent — AI-powered brand creation system.
Creates complete brand packages (brand.json + voice.json + logos) from scratch.
Strategy-first architecture: brand strategy feeds all downstream asset generation.
"""

import os
import json
import base64
import re
import socket
from pathlib import Path

import httpx
from google.adk.agents import Agent

# --- Config ---
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")
PERPLEXITY_API = "https://api.perplexity.ai/chat/completions"
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-exp-image-generation")
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"
OUTPUT_DIR = Path(__file__).parent / "output"

# --- Rate Guards ---
_perplexity_calls = 0
_perplexity_max = 10

_gemini_image_calls = 0
_gemini_image_max = 15

_google_search_calls = 0
_google_search_max = 10


def _perplexity_guard() -> dict | None:
    global _perplexity_calls
    _perplexity_calls += 1
    remaining = _perplexity_max - _perplexity_calls
    if _perplexity_calls > _perplexity_max:
        return {"error": f"Perplexity call limit reached ({_perplexity_max}). Work with data you have."}
    if remaining <= 2:
        print(f"[rate-guard] WARNING: {remaining} Perplexity calls remaining")
    return None


def _gemini_image_guard() -> dict | None:
    global _gemini_image_calls
    _gemini_image_calls += 1
    remaining = _gemini_image_max - _gemini_image_calls
    if _gemini_image_calls > _gemini_image_max:
        return {"error": f"Gemini Image call limit reached ({_gemini_image_max}). Work with logos you have."}
    if remaining <= 2:
        print(f"[rate-guard] WARNING: {remaining} Gemini Image calls remaining")
    return None


def _google_search_guard() -> dict | None:
    global _google_search_calls
    _google_search_calls += 1
    remaining = _google_search_max - _google_search_calls
    if _google_search_calls > _google_search_max:
        return {"error": f"Google Search call limit reached ({_google_search_max}). Work with data you have."}
    if remaining <= 2:
        print(f"[rate-guard] WARNING: {remaining} Google Search calls remaining")
    return None


# --- Color Helpers ---

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0,min(255,r)):02X}{max(0,min(255,g)):02X}{max(0,min(255,b)):02X}"


def _lighten(hex_color: str, factor: float = 0.3) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(int(r + (255 - r) * factor), int(g + (255 - g) * factor), int(b + (255 - b) * factor))


def _darken(hex_color: str, factor: float = 0.3) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(int(r * (1 - factor)), int(g * (1 - factor)), int(b * (1 - factor)))


# ============================================================
# TOOL 1: research_market
# ============================================================

def research_market(industry: str, target_audience: str, region: str = "global") -> dict:
    """Research market landscape, trends, and opportunities for a brand in a specific industry.
    Returns market intelligence including trends, audience insights, and positioning opportunities.

    Args:
        industry: The industry or market vertical (e.g., "sustainable fashion", "fintech", "health tech").
        target_audience: The target audience description (e.g., "Gen Z consumers", "young professionals").
        region: The geographic region to focus on (e.g., "UAE", "US", "global"). Default "global".
    """
    blocked = _perplexity_guard()
    if blocked:
        return blocked
    if not PERPLEXITY_API_KEY:
        return {"error": "PERPLEXITY_API_KEY not set in .env"}

    query = (
        f"Market analysis for {industry} targeting {target_audience} in {region}. "
        f"Include: 1) Current market trends and growth drivers, "
        f"2) Target audience preferences and behaviors, "
        f"3) Brand positioning opportunities and gaps in the market, "
        f"4) Key success factors for new brands in this space, "
        f"5) Common brand archetypes used by market leaders."
    )
    try:
        resp = httpx.post(
            PERPLEXITY_API,
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
            json={"model": "sonar-pro", "messages": [{"role": "user", "content": query}]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])
        return {
            "market_intelligence": answer,
            "citations": citations,
            "query_context": {"industry": industry, "target_audience": target_audience, "region": region},
        }
    except httpx.HTTPStatusError as e:
        return {"error": f"Perplexity API error: {e.response.status_code}", "detail": e.response.text[:500]}
    except Exception as e:
        return {"error": f"Market research failed: {str(e)}"}


# ============================================================
# TOOL 2: analyze_competitors
# ============================================================

def analyze_competitors(competitors: str, industry: str) -> dict:
    """Analyze competitor brands in an industry using Google Search.
    Returns competitor positioning, visual identity patterns, and messaging approaches.

    Args:
        competitors: Comma-separated list of competitor brand names (e.g., "Nike, Adidas, Puma").
        industry: The industry context for analysis (e.g., "athletic wear", "fintech").
    """
    blocked = _google_search_guard()
    if blocked:
        return blocked
    if not GOOGLE_SEARCH_API_KEY:
        return {"error": "GOOGLE_SEARCH_API_KEY not set in .env"}

    competitor_list = [c.strip() for c in competitors.split(",") if c.strip()]
    results = []

    for competitor in competitor_list[:5]:
        try:
            resp = httpx.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": GOOGLE_SEARCH_API_KEY,
                    "cx": GOOGLE_SEARCH_ENGINE_ID,
                    "q": f"{competitor} {industry} brand identity positioning tagline",
                    "num": 5,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            snippets = []
            for item in data.get("items", []):
                snippets.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                })
            results.append({"competitor": competitor, "findings": snippets})
        except Exception as e:
            results.append({"competitor": competitor, "error": str(e)})

    return {"industry": industry, "competitors_analyzed": len(results), "results": results}


# ============================================================
# TOOL 3: generate_brand_strategy
# ============================================================

def generate_brand_strategy(
    brand_name: str,
    industry: str,
    target_audience: str,
    values: str,
    mood: str,
    differentiator: str = "",
    market_research: str = "",
    competitor_insights: str = "",
) -> dict:
    """Generate a comprehensive brand strategy from user inputs and research data.
    This is a LOCAL computation — no external API calls. Structures all inputs into a cohesive strategy.

    Args:
        brand_name: The brand name.
        industry: The industry or vertical.
        target_audience: Description of the target audience.
        values: Comma-separated core brand values (e.g., "innovation, trust, sustainability").
        mood: The desired brand mood/aesthetic (e.g., "premium minimalist", "bold and energetic").
        differentiator: What makes this brand unique vs competitors. Optional.
        market_research: Summary of market research findings (from research_market). Optional.
        competitor_insights: Summary of competitor analysis (from analyze_competitors). Optional.
    """
    values_list = [v.strip() for v in values.split(",") if v.strip()]

    mood_archetypes = {
        "premium": "Ruler / Sage",
        "minimalist": "Sage / Creator",
        "bold": "Hero / Rebel",
        "energetic": "Explorer / Magician",
        "playful": "Jester / Innocent",
        "trustworthy": "Caregiver / Sage",
        "innovative": "Creator / Magician",
        "luxury": "Ruler / Lover",
        "sustainable": "Caregiver / Creator",
        "tech": "Magician / Creator",
        "warm": "Caregiver / Everyman",
        "edgy": "Rebel / Outlaw",
        "clean": "Innocent / Sage",
        "professional": "Sage / Ruler",
    }

    suggested_archetype = "Creator"
    mood_lower = mood.lower()
    for keyword, archetype in mood_archetypes.items():
        if keyword in mood_lower:
            suggested_archetype = archetype
            break

    strategy = {
        "brand_name": brand_name,
        "industry": industry,
        "target_audience": target_audience,
        "core_values": values_list,
        "mood": mood,
        "suggested_archetype": suggested_archetype,
        "differentiator": differentiator or "To be defined based on research",
        "positioning_statement": (
            f"{brand_name} is a {mood} {industry} brand for {target_audience}, "
            f"built on {', '.join(values_list[:3])}."
        ),
        "brand_promise": f"We help {target_audience} through {', '.join(values_list[:2])}.",
        "market_context": market_research[:500] if market_research else "No market research provided yet.",
        "competitive_landscape": competitor_insights[:500] if competitor_insights else "No competitor analysis provided yet.",
        "next_steps": [
            "Confirm archetype and positioning",
            "Generate color palette based on mood and industry",
            "Select typography that matches the brand personality",
            "Generate logo variations",
            "Create voice configuration",
            "Assemble final brand.json",
        ],
    }

    return {"strategy": strategy, "status": "ready_for_visual_identity"}


# ============================================================
# TOOL 4: generate_color_palette
# ============================================================

def generate_color_palette(
    primary_hex: str,
    accent_hex: str,
    mood: str = "modern",
    industry: str = "",
    dark_mode: bool = True,
) -> dict:
    """Generate a complete 11-color brand palette from primary and accent colors.
    Derives all required colors (primaryLight, text, bg, divider, shadow) to match the Content-Agent schema.

    Args:
        primary_hex: Primary brand color as hex (e.g., "#5A279B").
        accent_hex: Accent/secondary brand color as hex (e.g., "#7364D2").
        mood: Visual mood — "modern", "classic", "playful", or "minimal". Default "modern".
        industry: Industry context for color psychology guidance. Optional.
        dark_mode: Whether to optimize for dark mode UI. Default True.
    """
    hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
    if not hex_pattern.match(primary_hex):
        return {"error": f"Invalid primary hex: {primary_hex}. Must be #RRGGBB format."}
    if not hex_pattern.match(accent_hex):
        return {"error": f"Invalid accent hex: {accent_hex}. Must be #RRGGBB format."}

    primary_light = _lighten(primary_hex, 0.35)
    pr, pg, pb = _hex_to_rgb(primary_hex)

    if dark_mode:
        colors = {
            "primary": primary_hex,
            "primaryLight": primary_light,
            "accent": accent_hex,
            "textDark": "#FFFFFF",
            "textMedium": "#A0A4AA",
            "textLight": "#6B7280",
            "bgLight": "#F7F8FD",
            "bgDark": "#0F0F0F",
            "bgOverlay": "rgba(15, 15, 15, 0.95)",
            "divider": "#1E1E1E",
            "shadow": f"rgba({pr}, {pg}, {pb}, 0.2)",
        }
    else:
        colors = {
            "primary": primary_hex,
            "primaryLight": primary_light,
            "accent": accent_hex,
            "textDark": "#111827",
            "textMedium": "#6B7280",
            "textLight": "#9CA3AF",
            "bgLight": "#FFFFFF",
            "bgDark": "#111827",
            "bgOverlay": "rgba(17, 24, 39, 0.95)",
            "divider": "#E5E7EB",
            "shadow": f"rgba({pr}, {pg}, {pb}, 0.15)",
        }

    return {
        "colors": colors,
        "mood": mood,
        "dark_mode": dark_mode,
        "gradient": f"linear-gradient(180deg, {primary_hex}, {accent_hex})",
        "note": "All 11 required color keys present. Ready for assemble_brand_json.",
    }


# ============================================================
# TOOL 5: generate_typography
# ============================================================

def generate_typography(
    heading_font: str = "",
    body_font: str = "",
    mono_font: str = "",
    style: str = "modern",
) -> dict:
    """Generate a complete typography system including font stacks and size scale.
    Produces fonts, spacing, borderRadius, and typography objects matching Content-Agent schema.

    Args:
        heading_font: Google Font name for headings (e.g., "Inter", "Playfair Display"). Empty for auto-select based on style.
        body_font: Google Font name for body text. Empty to match heading font.
        mono_font: Monospace font name. Empty for default.
        style: Typography style preset — "modern", "classic", "playful", or "minimal". Default "modern".
    """
    presets = {
        "modern": {
            "heading_default": "Inter",
            "body_default": "Inter",
            "mono_default": "JetBrains Mono",
            "h1": {"size": 88, "weight": 700, "transform": "none"},
            "h2": {"size": 64, "weight": 700, "transform": "none"},
            "h3": {"size": 48, "weight": 600},
            "body_style": {"size": 40, "weight": 400},
            "label": {"size": 32, "weight": 500, "letterSpacing": 1},
            "stat": {"size": 64, "weight": 700, "font": "mono"},
            "spacing": {"xs": 8, "sm": 16, "md": 24, "lg": 48, "xl": 80, "xxl": 120},
            "borderRadius": {"sm": 8, "md": 12, "lg": 20},
        },
        "classic": {
            "heading_default": "Playfair Display",
            "body_default": "Source Serif Pro",
            "mono_default": "IBM Plex Mono",
            "h1": {"size": 80, "weight": 700, "transform": "none"},
            "h2": {"size": 60, "weight": 600, "transform": "none"},
            "h3": {"size": 44, "weight": 600},
            "body_style": {"size": 40, "weight": 400},
            "label": {"size": 30, "weight": 500, "letterSpacing": 2},
            "stat": {"size": 60, "weight": 700, "font": "mono"},
            "spacing": {"xs": 8, "sm": 16, "md": 24, "lg": 48, "xl": 80, "xxl": 120},
            "borderRadius": {"sm": 4, "md": 8, "lg": 12},
        },
        "playful": {
            "heading_default": "Outfit",
            "body_default": "Nunito",
            "mono_default": "Fira Code",
            "h1": {"size": 84, "weight": 800, "transform": "none"},
            "h2": {"size": 64, "weight": 700, "transform": "none"},
            "h3": {"size": 48, "weight": 600},
            "body_style": {"size": 42, "weight": 400},
            "label": {"size": 32, "weight": 600, "letterSpacing": 0},
            "stat": {"size": 68, "weight": 800, "font": "mono"},
            "spacing": {"xs": 10, "sm": 18, "md": 28, "lg": 52, "xl": 88, "xxl": 130},
            "borderRadius": {"sm": 12, "md": 20, "lg": 32},
        },
        "minimal": {
            "heading_default": "DM Sans",
            "body_default": "DM Sans",
            "mono_default": "DM Mono",
            "h1": {"size": 72, "weight": 500, "transform": "none", "letterSpacing": -1},
            "h2": {"size": 56, "weight": 500, "transform": "none"},
            "h3": {"size": 40, "weight": 500},
            "body_style": {"size": 36, "weight": 400},
            "label": {"size": 28, "weight": 400, "letterSpacing": 3, "transform": "uppercase"},
            "stat": {"size": 56, "weight": 500, "font": "mono"},
            "spacing": {"xs": 6, "sm": 12, "md": 20, "lg": 40, "xl": 72, "xxl": 108},
            "borderRadius": {"sm": 2, "md": 4, "lg": 8},
        },
    }

    preset = presets.get(style, presets["modern"])

    h_font = heading_font or preset["heading_default"]
    b_font = body_font or preset["body_default"]
    m_font = mono_font or preset["mono_default"]

    fonts = {
        "heading": f"{h_font}, ui-sans-serif, system-ui, sans-serif",
        "body": f"{b_font}, ui-sans-serif, system-ui, sans-serif",
        "mono": f"{m_font}, ui-monospace, monospace",
    }

    typography = {
        "h1": preset["h1"],
        "h2": preset["h2"],
        "h3": preset["h3"],
        "body": preset["body_style"],
        "label": preset["label"],
        "stat": preset["stat"],
    }

    return {
        "fonts": fonts,
        "typography": typography,
        "spacing": preset["spacing"],
        "borderRadius": preset["borderRadius"],
        "style_preset": style,
        "note": "All schema-required keys present. Ready for assemble_brand_json.",
    }


# ============================================================
# TOOL 6: generate_logo
# ============================================================

def generate_logo(
    brand_name: str,
    style_description: str,
    primary_color: str,
    accent_color: str = "",
    background: str = "transparent",
    variation: str = "primary",
) -> dict:
    """Generate a logo using the Gemini Image API. Returns base64-encoded PNG image data.
    Call multiple times with different 'variation' values for a complete logo set.

    Args:
        brand_name: The brand name to feature in/inspire the logo.
        style_description: Description of desired logo style (e.g., "minimalist geometric monogram", "bold wordmark").
        primary_color: Primary brand color as hex (e.g., "#5A279B").
        accent_color: Optional accent color as hex. Empty for monochromatic.
        background: Background — "transparent", "dark", "light", or a hex code. Default "transparent".
        variation: Logo type — "primary" (full logo), "monogram" (icon only), "wordmark" (text only). Default "primary".
    """
    blocked = _gemini_image_guard()
    if blocked:
        return blocked
    if not GOOGLE_API_KEY:
        return {"error": "GOOGLE_API_KEY not set in .env"}

    bg_desc = {
        "transparent": "on a clean white background",
        "dark": "on a dark black (#0F0F0F) background",
        "light": "on a clean white (#FFFFFF) background",
    }.get(background, f"on a {background} colored background")

    variation_desc = {
        "primary": f"Full logo for '{brand_name}' — include both a symbol/icon and the brand name text",
        "monogram": f"Monogram/icon-only logo for '{brand_name}' — a single distinctive symbol or letter mark, NO text",
        "wordmark": f"Wordmark logo for '{brand_name}' — stylized text of the brand name only, clean typography",
    }.get(variation, f"Logo for '{brand_name}'")

    color_desc = f"using {primary_color} as the primary color"
    if accent_color:
        color_desc += f" and {accent_color} as accent"

    prompt = (
        f"Generate a professional, high-quality vector-style logo. "
        f"{variation_desc}. "
        f"Style: {style_description}. "
        f"Colors: {color_desc}. "
        f"Placement: {bg_desc}. "
        f"Requirements: Clean lines, scalable design, professional quality, minimal detail. "
        f"No gradients unless specified. No photographic elements. No watermarks. "
        f"Suitable for both digital and print use."
    )

    try:
        resp = httpx.post(
            f"{GEMINI_API}/models/{GEMINI_IMAGE_MODEL}:generateContent?key={GOOGLE_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return {"error": "No image generated. Check model availability.", "raw": str(data)[:500]}

        parts = candidates[0].get("content", {}).get("parts", [])
        image_data = None
        text_response = ""

        for part in parts:
            if "inlineData" in part:
                image_data = part["inlineData"]["data"]
            elif "text" in part:
                text_response = part["text"]

        if not image_data:
            return {"error": "No image in response", "text": text_response, "raw": str(data)[:500]}

        return {
            "image_base64": image_data,
            "variation": variation,
            "brand_name": brand_name,
            "prompt_used": prompt[:200],
            "description": text_response[:300] if text_response else f"{variation} logo for {brand_name}",
            "note": "Use save_brand_package to write this to disk.",
        }
    except httpx.HTTPStatusError as e:
        return {"error": f"Gemini Image API error: {e.response.status_code}", "detail": e.response.text[:500]}
    except Exception as e:
        return {"error": f"Logo generation failed: {str(e)}"}


# ============================================================
# TOOL 7: generate_voice_config
# ============================================================

def generate_voice_config(
    personality_traits: str,
    voice_description: str,
    writing_rules: str,
    tts_voice_name: str = "Kore",
    tone_spectrum: str = "",
    voice_test: str = "",
) -> dict:
    """Generate a complete voice.json configuration for the brand.
    Includes TTS settings, writing rules, and personality definition.

    Args:
        personality_traits: Comma-separated personality traits (e.g., "Professional, Bold, Innovative"). Minimum 3.
        voice_description: One-sentence description of the brand voice (e.g., "Confident advisor who keeps it real").
        writing_rules: Pipe-separated writing rules (use | delimiter). Minimum 5 rules.
        tts_voice_name: Gemini TTS voice name. Options: Kore, Charon, Fenrir, Aoede, Puck, Leda. Default "Kore".
        tone_spectrum: Comma-separated key:value pairs (e.g., "professional:0.8,bold:0.7,salesy:0.0"). Optional — auto-generated from traits if empty.
        voice_test: Question to test if content matches the voice (e.g., "Would a Gen Z reader share this?"). Optional.
    """
    traits = [t.strip() for t in personality_traits.split(",") if t.strip()]
    if len(traits) < 3:
        return {"error": f"Need at least 3 personality traits, got {len(traits)}."}

    rules = [r.strip() for r in writing_rules.split("|") if r.strip()]
    if len(rules) < 5:
        return {"error": f"Need at least 5 writing rules, got {len(rules)}. Separate with | character."}

    if tone_spectrum:
        spectrum = {}
        for pair in tone_spectrum.split(","):
            if ":" in pair:
                key, val = pair.split(":", 1)
                try:
                    spectrum[key.strip()] = round(float(val.strip()), 2)
                except ValueError:
                    spectrum[key.strip()] = 0.5
    else:
        spectrum = {trait.lower(): 0.8 for trait in traits}
        spectrum["salesy"] = 0.0
        spectrum["corporate"] = 0.15

    voice_json = {
        "tts": {
            "provider": "gemini",
            "model": "gemini-2.5-flash-preview-tts",
            "voiceName": tts_voice_name,
            "description": voice_description,
            "settings": {"speakingRate": 1.0, "pitch": 0},
        },
        "writingRules": rules,
    }

    brand_voice = {
        "personality": traits,
        "description": voice_description,
        "toneSpectrum": spectrum,
        "voiceTest": voice_test or f"Does this sound like a {traits[0].lower()}, {traits[1].lower()} brand?",
    }

    return {
        "voice_json": voice_json,
        "brand_voice_section": brand_voice,
        "note": "voice_json -> save as voice.json. brand_voice_section -> use in assemble_brand_json voice field.",
    }


# ============================================================
# TOOL 8: assemble_brand_json
# ============================================================

def assemble_brand_json(
    name: str,
    description: str,
    website: str,
    tagline: str,
    cta: str,
    archetype: str,
    colors_json: str,
    fonts_json: str,
    spacing_json: str,
    border_radius_json: str,
    typography_json: str,
    voice_json: str,
    campaign_lines: str,
    logo_path: str = "assets/logo-primary.png",
    video_bg_color: str = "",
) -> dict:
    """Assemble all brand components into a complete brand.json. Validates against Content-Agent schema.
    All JSON arguments should be stringified JSON objects from previous tool outputs.

    Args:
        name: Brand name.
        description: Brand description (1-2 sentences).
        website: Brand website URL (must include https://).
        tagline: Brand tagline.
        cta: Primary call-to-action text (e.g., "Get Started", "Learn More").
        archetype: Brand archetype (e.g., "Creator / Magician").
        colors_json: JSON string of colors object (the "colors" key from generate_color_palette).
        fonts_json: JSON string of fonts object (the "fonts" key from generate_typography).
        spacing_json: JSON string of spacing object (from generate_typography).
        border_radius_json: JSON string of borderRadius object (from generate_typography).
        typography_json: JSON string of typography object (the "typography" key from generate_typography).
        voice_json: JSON string of voice object (the "brand_voice_section" key from generate_voice_config).
        campaign_lines: Pipe-separated campaign lines (minimum 5). Use | as delimiter.
        logo_path: Relative path to primary logo file. Default "assets/logo-primary.png".
        video_bg_color: Background color for video defaults. Empty to use bgDark from colors.
    """
    try:
        colors = json.loads(colors_json) if isinstance(colors_json, str) else colors_json
        fonts = json.loads(fonts_json) if isinstance(fonts_json, str) else fonts_json
        spacing = json.loads(spacing_json) if isinstance(spacing_json, str) else spacing_json
        border_radius = json.loads(border_radius_json) if isinstance(border_radius_json, str) else border_radius_json
        typography = json.loads(typography_json) if isinstance(typography_json, str) else typography_json
        voice = json.loads(voice_json) if isinstance(voice_json, str) else voice_json
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in input: {str(e)}"}

    lines = [l.strip() for l in campaign_lines.split("|") if l.strip()]
    if len(lines) < 5:
        return {"error": f"Need at least 5 campaign lines, got {len(lines)}. Separate with | character."}

    required_colors = [
        "primary", "primaryLight", "accent", "textDark", "textMedium",
        "textLight", "bgLight", "bgDark", "bgOverlay", "divider", "shadow",
    ]
    missing_colors = [k for k in required_colors if k not in colors]
    if missing_colors:
        return {"error": f"Missing required color keys: {missing_colors}"}

    required_typo = ["h1", "h2", "h3", "body", "label", "stat"]
    missing_typo = [k for k in required_typo if k not in typography]
    if missing_typo:
        return {"error": f"Missing required typography keys: {missing_typo}"}

    if not website.startswith("http"):
        website = f"https://{website}"

    bg_color = video_bg_color or colors.get("bgDark", "#0F0F0F")
    primary_color = colors.get("primary", "#000000")

    brand = {
        "name": name,
        "description": description,
        "version": "1.0.0",
        "website": website,
        "tagline": tagline,
        "cta": cta,
        "archetype": archetype,
        "colors": colors,
        "fonts": fonts,
        "spacing": spacing,
        "borderRadius": border_radius,
        "typography": typography,
        "assets": {"logo": logo_path},
        "voice": voice,
        "campaignLines": lines,
        "videoDefaults": {
            "aspectRatio": "9:16",
            "resolution": "1080x1920",
            "bgColor": bg_color,
            "progressBarColor": primary_color,
        },
    }

    return {
        "brand_json": brand,
        "validation": "passed",
        "field_count": len(brand),
        "color_count": len(colors),
        "campaign_line_count": len(lines),
        "note": "Schema-validated. Ready for save_brand_package.",
    }


# ============================================================
# TOOL 9: check_domain
# ============================================================

def check_domain(domain: str) -> dict:
    """Check if a domain name is likely taken by performing a DNS lookup.
    A successful DNS resolution means the domain IS registered/taken.

    Args:
        domain: The domain to check (e.g., "example.com"). Do not include http/https.
    """
    domain = domain.strip().lower()
    if domain.startswith("http"):
        domain = domain.split("//", 1)[-1].split("/")[0]

    try:
        socket.getaddrinfo(domain, None, socket.AF_INET)
        return {"domain": domain, "likely_taken": True, "note": "DNS resolves — domain is registered."}
    except socket.gaierror:
        return {"domain": domain, "likely_taken": False, "note": "DNS does not resolve — domain may be available."}
    except Exception as e:
        return {"domain": domain, "likely_taken": None, "error": f"Could not check: {str(e)}"}


# ============================================================
# TOOL 10: save_brand_package
# ============================================================

def save_brand_package(
    brand_slug: str,
    brand_json_str: str,
    voice_json_str: str,
    logo_images: str = "",
) -> dict:
    """Save the complete brand package to disk: brand.json, voice.json, and logo images.

    Args:
        brand_slug: URL-safe brand identifier (e.g., "my-brand"). Used as directory name.
        brand_json_str: Complete brand.json as a JSON string (the "brand_json" key from assemble_brand_json).
        voice_json_str: Complete voice.json as a JSON string (the "voice_json" key from generate_voice_config).
        logo_images: JSON string mapping filenames to base64 data (e.g., '{"logo-primary.png": "iVBOR..."}').  Empty if no logos.
    """
    slug = re.sub(r"[^a-z0-9-]", "", brand_slug.lower().replace(" ", "-"))
    if not slug:
        return {"error": "Invalid brand_slug — must contain letters or numbers."}

    brand_dir = OUTPUT_DIR / slug
    assets_dir = brand_dir / "assets"

    try:
        brand_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Write brand.json
        try:
            brand_data = json.loads(brand_json_str) if isinstance(brand_json_str, str) else brand_json_str
        except json.JSONDecodeError as e:
            return {"error": f"Invalid brand_json_str: {str(e)}"}

        brand_path = brand_dir / "brand.json"
        brand_path.write_text(json.dumps(brand_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Write voice.json
        try:
            voice_data = json.loads(voice_json_str) if isinstance(voice_json_str, str) else voice_json_str
        except json.JSONDecodeError as e:
            return {"error": f"Invalid voice_json_str: {str(e)}"}

        voice_path = brand_dir / "voice.json"
        voice_path.write_text(json.dumps(voice_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Save logo images
        saved_logos = []
        if logo_images:
            try:
                logos = json.loads(logo_images) if isinstance(logo_images, str) else logo_images
            except json.JSONDecodeError:
                logos = {}

            for filename, b64_data in logos.items():
                try:
                    img_bytes = base64.b64decode(b64_data)
                    img_path = assets_dir / filename
                    img_path.write_bytes(img_bytes)
                    saved_logos.append(str(img_path))
                except Exception as e:
                    saved_logos.append(f"FAILED {filename}: {str(e)}")

        return {
            "saved": True,
            "directory": str(brand_dir),
            "files": {
                "brand_json": str(brand_path),
                "voice_json": str(voice_path),
                "logos": saved_logos,
            },
            "note": f"Brand package saved to {brand_dir}. Copy to Content-Agent brands/{slug}/ when ready.",
        }
    except Exception as e:
        return {"error": f"Failed to save brand package: {str(e)}"}


# ============================================================
# AGENT DEFINITION
# ============================================================

root_agent = Agent(
    model="gemini-3-pro-preview",
    name="brand_agent",
    description="AI-powered brand creation system. Creates complete brand packages from scratch.",
    instruction="""You are a senior brand strategist and designer. You create complete brand packages — strategy, visual identity, voice, and configuration files — through a structured 6-phase workflow.

## YOUR WORKFLOW (follow strictly in order)

### PHASE 1: DISCOVERY
Ask the user for:
- Brand name (or help them brainstorm one)
- Industry / market vertical
- Target audience
- Core values (3-5 words)
- Desired mood/aesthetic (e.g., "premium minimalist", "bold tech", "warm organic")
- Key competitors (2-3 names)
- Any color preferences
- Region/market

Do NOT proceed until you have at least: brand name, industry, target audience, values, and mood.

### PHASE 2: RESEARCH
Use these tools:
- `research_market` — get market intelligence and positioning opportunities
- `analyze_competitors` — study competitor brand approaches

Present findings to the user. Ask for confirmation before proceeding.

### PHASE 3: STRATEGY
Use these tools:
- `generate_brand_strategy` — structure everything into a cohesive strategy
- `check_domain` — verify domain availability for the brand name

Present the strategy (archetype, positioning, brand promise). Ask for confirmation.

### PHASE 4: VISUAL IDENTITY
Use these tools:
- `generate_color_palette` — create the complete 11-color system
- `generate_typography` — select fonts and type scale
- `generate_logo` — generate 2-3 logo variations (primary, monogram, wordmark)

Present visual choices. Ask for confirmation (user may want color/font adjustments).

### PHASE 5: CONFIGURATION
Use these tools:
- `generate_voice_config` — create voice.json with TTS settings and writing rules
- `assemble_brand_json` — compile everything into the final brand.json

Show the user a summary of the assembled brand package. Ask for final approval.

### PHASE 6: SAVE
Use this tool:
- `save_brand_package` — write all files to disk

Report where files were saved and suggest next steps.

## CRITICAL RULES

1. **Strategy first** — NEVER generate colors, fonts, or logos before completing strategy (Phase 3)
2. **User confirmation between phases** — Always present results and ask "Shall I proceed to [next phase]?" before moving on
3. **Minimum counts** — At least 5 campaign lines, 5 writing rules, 3 personality traits
4. **Schema compliance** — The output brand.json MUST have all 11 color keys, all 6 typography keys, spacing, borderRadius, assets, voice, campaignLines, and videoDefaults
5. **Rate awareness** — You have limited API calls (10 Perplexity, 15 Gemini Image, 10 Google Search). Be strategic.
6. **Colors as hex** — Always use #RRGGBB format for colors. Validate before generating.
7. **Logo variations** — Generate at minimum: one "primary" (full logo) and one "monogram" (icon only)
8. **JSON string arguments** — When calling assemble_brand_json or save_brand_package, pass JSON objects as stringified JSON
9. **Campaign lines** — Write at least 5 varied lines: mix taglines, data points, CTAs, and value propositions

## PASSING DATA BETWEEN TOOLS

When calling `assemble_brand_json`, extract the specific keys from previous tool outputs:
- colors_json = the "colors" object from generate_color_palette result
- fonts_json = the "fonts" object from generate_typography result
- spacing_json = the "spacing" object from generate_typography result
- border_radius_json = the "borderRadius" object from generate_typography result
- typography_json = the "typography" object from generate_typography result
- voice_json = the "brand_voice_section" object from generate_voice_config result

When calling `save_brand_package`:
- brand_json_str = the "brand_json" object from assemble_brand_json result
- voice_json_str = the "voice_json" object from generate_voice_config result
- logo_images = JSON object mapping filenames to base64 strings from generate_logo results

## REFERENCE: REVONA BRAND (Gold Standard)

The Revona brand is the quality benchmark:
- Archetype: "Expert-Empowerer hybrid"
- Colors: Premium violet palette with dark mode default, 11+ color keys
- Voice: Professional, empowering, trustworthy — anti-corporate tone
- Campaign lines: 12 varied lines mixing taglines, data points, and CTAs
- Typography: Clean font family, proper scale from 34px labels to 88px h1
- Writing rules: 12 specific rules including do's and don'ts

Your output should match this level of quality and completeness.""",
    tools=[
        research_market,
        analyze_competitors,
        generate_brand_strategy,
        generate_color_palette,
        generate_typography,
        generate_logo,
        generate_voice_config,
        assemble_brand_json,
        check_domain,
        save_brand_package,
    ],
)
