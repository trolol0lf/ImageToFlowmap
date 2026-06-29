flowshader = """
#version 330 core

in vec2 vTex;
out vec4 FragColor;

uniform sampler2D uTex;
uniform sampler2D uFlowmap;

uniform float uOffset;
uniform float uAlpha;
uniform vec2  uPanning;
uniform bool  uSelected;
uniform float uTimeUnscaled;
uniform float uTime;
uniform vec4  uColor;

uniform int   uDrawAsColor;
// 0 = Flowing
// 1 = Flowmap RGB
// 2 = Flow Distortion Debug
// 3 = Heightmap Debug

uniform int   uFlowType;
// 0 = Water
// 1 = Lava
// 2 = Fog

uniform vec3 uMaskX;
uniform vec3 uMaskY;

uniform float uFlowStrength;
uniform float uWaveFreq;
uniform float uWaveSharpness;
uniform float uFoamThreshold;
uniform float uWaveScale;

// ---------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------
#define PI 3.14159265359
#define GOLDEN_ANGLE 2.39996323
#define NUM_OCTAVES 6

// ---------------------------------------------------------------------
// hashing / pseudo random
// ---------------------------------------------------------------------
float hash11(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// ---------------------------------------------------------------------
// flowmap decoding
// uMaskX / uMaskY pick the right channels (e.g. (1,0,0)/(0,1,0)).
// 0.5 in the texture == "no flow", alpha carries per-pixel influence so
// several flowmap layers can be stacked without stomping on each other.
//
// sampleFlow pre-warps the UV with a Hermite curve before the hardware's
// bilinear fetch -> softens the visible texel grid of low-res flowmaps
// without needing any extra samples ("continuous resolve").
// ---------------------------------------------------------------------
vec4 sampleFlow(vec2 uv)
{
    vec2 texSize  = vec2(textureSize(uFlowmap, 0));
    vec2 texel    = uv * texSize - 0.5;
    vec2 f        = fract(texel);
    vec2 f2       = f * f * (3.0 - 2.0 * f); // Hermite -> smooths the interpolation curve
    vec2 smoothUV = (floor(texel) + 0.5 + f2) / texSize;
    return texture(uFlowmap, smoothUV);
}

vec2 decodeFlowDir(vec4 flowSample)
{
    vec2 dir = (vec2(dot(flowSample.rgb, uMaskX), dot(flowSample.rgb, uMaskY)) - 0.5) * 2.0;
    if (length(dir) < 0.000001)
    {
        dir = vec2(0.0);
    }
    return dir;
}

// ---------------------------------------------------------------------
// multi-octave pseudo-Gerstner / FFT-style height field.
// A literal per-pixel FFT isn't practical here, so we fake the look of an
// FFT-ocean spectrum by summing a handful of sinusoids whose direction,
// frequency and phase come from a hash sequence placed on the golden
// angle. Directions stay chaotic/golden-angle distributed (so some
// octaves naturally run opposite the current) and only get *steered*
// towards the flow direction -- strongly for the big low-frequency
// rollers, barely at all for the fine high-frequency chop. That keeps
// the field irrational / non-repeating over a large area and avoids the
// "everything slides the same way" bias of a fully flow-aligned field.
// ---------------------------------------------------------------------
void computeWaveField(vec2 uv, vec2 flowDir, float freqMul, float speedMul, float time, float seed,
                       out float height, out vec2 grad, out float ampSum)
{
    height = 0.0;
    grad   = vec2(0.0);
    ampSum = 0.0;

    float amp  = 1.0;
    float freq = uWaveFreq * freqMul;

    float dirLen  = length(flowDir);
    vec2  flowN   = flowDir;
    float flowAmt = uFlowStrength;

    for (int i = 0; i < NUM_OCTAVES; i++)
    {
        float fi = float(i);

        // chaotic base direction, golden-angle distributed -> covers the
        // whole circle over NUM_OCTAVES, including directions roughly
        // opposing the flow
        float angle   = (fi + seed * 14.13) * GOLDEN_ANGLE;
        vec2  baseDir = vec2(cos(angle), sin(angle));

        // low-frequency rollers follow the current strongly; high-frequency
        // chop stays mostly chaotic/counter-propagating
        float octaveFalloff = mix(1.0, 0.3, fi / float(NUM_OCTAVES - 1));
        float steerAmt = flowAmt * 0.8 * octaveFalloff - 0.5;

        vec2 dir = normalize(mix(baseDir, flowN, 0));

        float phase  = hash11(fi * 13.17 + seed * 7.31) * 2.0 * PI;
        float phase2 = phase + hash11(fi * 9.91 + seed * 2.11) * 1.2;
        float spdBase = speedMul * (0.55 + 0.45 * hash11(fi * 3.31 + seed * 1.7));

        float x = dot(dir, uv) * freq;
        float t = time * spdBase * freq;

        // two counter-running phases superposed (standing-wave look) plus
        // slight asymmetry so it doesn't read as perfectly mirrored
        float asym = 0.88 + 0.12 * hash11(fi * 5.33 + seed * 6.77);
        float w1 = cos(x - t + phase);
        float w2 = cos(x + t + phase2);
        float wave = w1 + asym * w2;

        // analytic AA: fade this octave out before its phase-change-per-
        // pixel gets too fast to rasterize cleanly. NOTE: x already has
        // freq baked in, so we measure fwidth(x), not fwidth(x*freq) --
        // squaring freq here was what caused the "Lücken" at high
        // freq/scale values.
        float phaseRate = fwidth(x);
        float aa = 1.0 - smoothstep(1.5, 3.5, phaseRate);
        float ampAA = amp * aa;

        height += ampAA * wave;
        grad   += ampAA * freq * dir * (sin(x - t + phase) - asym * sin(x + t + phase2));
        ampSum += ampAA;

        freq *= 1.37;
        amp  *= 0.62;
    }

    grad = -grad;
}

// sharpen crests / flatten troughs the way real Gerstner waves look
float shapeCrests(float normHeight, float sharpness)
{
    float s = max(sharpness, 0.001);
    return sign(normHeight) * pow(abs(normHeight), 1.0 / s);
}

// ---------------------------------------------------------------------
// per-type "viscosity": water / lava / fog behave very differently
// ---------------------------------------------------------------------
void flowTypeParams(out float speedMul, out float freqMul, out float distortMul, out bool foamy, out vec3 tint)
{
    if (uFlowType == 1) // Lava: slow, thick, big rolling blobs, glow instead of foam
    {
        speedMul   = 0.22;
        freqMul    = 0.55;
        distortMul = 0.55;
        foamy      = false;
        tint       = vec3(1.0, 0.45, 0.12);
    }
    else if (uFlowType == 2) // Fog: very slow, very soft, barely any hard crest
    {
        speedMul   = 0.10;
        freqMul    = 0.28;
        distortMul = 0.35;
        foamy      = false;
        tint       = vec3(0.85, 0.88, 0.92);
    }
    else // Water: fast, sharp, foamy
    {
        speedMul   = 1.0;
        freqMul    = 1.0;
        distortMul = 1.0;
        foamy      = true;
        tint       = vec3(0.8, 0.95, 1.0);
    }
}

// ---------------------------------------------------------------------
// "blingbling" overlay for the selected flowmap layer
// ---------------------------------------------------------------------
vec3 selectionSparkle(vec2 uv, float time)
{
    return vec3(sin(time) * 0.35);
}

vec3 clampvec3(vec3 toclamp, float lowerborder, float upperborder)
{
    return vec3(clamp(toclamp.r, lowerborder, upperborder), clamp(toclamp.g, lowerborder, upperborder), clamp(toclamp.b, lowerborder, upperborder));
}

void main()
{
    vec4 flowSample = sampleFlow(vTex);
    vec3  outColor = vec3(0.0);
    float outAlpha = uAlpha;
    vec2 flowDir = decodeFlowDir(flowSample);

    if (uDrawAsColor == 1)
    {
        // ---- Flowmap RGB: raw flowmap, alpha respected ----
        outColor = flowSample.rgb;
    }
    else
    {
        float speedMul, freqMul, distortMul;
        bool foamy;
        vec3 typeTint;
        flowTypeParams(speedMul, freqMul, distortMul, foamy, typeTint);

        // gates ALL motion (panning, traveling phase) so FlowStrength == 0
        // really means standstill, in every draw mode -- not just a frozen
        // chaotic ripple shape drifting off in some constant direction
        float flowGate = clamp(uFlowStrength, 0.0, 1.0);

        float seed = uOffset * 17.0 + 0.123;
        float time = (uTime * 10.0 + uOffset * 4.0) * flowGate; // desync stacked layers using the same shader
        vec2  panOffset = flowDir * uTime;

        // classic dual-phase flowmap advection (Valve-style) so the UV offset
        // never "snaps" when it wraps around
        float phase0 = fract(time * 0.35);

        // wave field evaluated on the un-advected base uv, so the chop reads as
        // belonging to the surface instead of getting dragged along with it
        float height; vec2 grad; float ampSum;
        computeWaveField((vTex + panOffset) * uWaveScale, flowDir, freqMul, speedMul, time, seed,
                        height, grad, ampSum);

        float normHeight = height / max(ampSum, 1e-4);
        vec2  normGrad   = grad   / max(ampSum, 1e-4);
        float gradMag    = length(normGrad);
        float shaped     = shapeCrests(normHeight, uWaveSharpness);

        if (uDrawAsColor == 2)
        {
            // ---- Flow Distortion Debug: grid bent by the flow + wave field ----
            vec2 distUV = vTex + panOffset - flowDir * phase0 * 0.08 * distortMul
                                + normGrad * flowGate * 0.02;
            vec2 gridUV = distUV * 24.0;
            vec2 gridF  = abs(fract(gridUV) - 0.5);
            float lineW = fwidth(gridUV.x) * 1.5 + 0.001;
            float line  = 1.0 - smoothstep(0.0, lineW, min(gridF.x, gridF.y) - (0.5 - lineW));

            // faint directional tint so the flow direction itself stays legible
            vec3 dirCol = vec3(0.5) + vec3(flowDir, 0.0) * 0.5;
            outColor  = mix(dirCol * 0.35, vec3(1.0, 0.85, 0.2), line);
            outAlpha *= mix(0.35, 1.0, line);
        }
        else if (uDrawAsColor == 3)
        {
            // ---- Heightmap Debug: STATIC contour lines of the flowmap's
            // own strength field -- no time, no generated waves, so it
            // stays a clean, motionless reference view of the flowmap ----
            float h01 = clamp(length(flowDir), 0.0, 1.0);

            vec3 low  = vec3(0.05, 0.05, 0.25);
            vec3 mid  = vec3(0.10, 0.55, 0.65);
            vec3 high = vec3(1.00, 0.95, 0.70);
            vec3 ramp = (h01 < 0.5) ? mix(low, mid, h01 * 2.0)
                                    : mix(mid, high, (h01 - 0.5) * 2.0);

            float levels  = 14.0;
            float band    = fract(h01 * levels);
            float lineW2  = fwidth(h01 * levels) * 1.5 + 0.0008;
            float contour = 1.0 - smoothstep(0.0, lineW2, min(band, 1.0 - band));

            outColor = mix(ramp, vec3(0.0), contour * 0.6);
        }
        else if (uDrawAsColor == 0)
        {
            // ---- Flowing: the actual rendered surface ----
            // procedural body color (depth-shaded by height) instead of a
            // sampled texture -- no particles, just the wave field itself
            vec3 base = typeTint * (0.16 + 0.12 * normHeight);

            // no real light/perspective here, so fake shading from the wave
            // gradient instead: steep slopes catch a glint, like sun-glitter
            float glint  = pow(clamp(gradMag, 0.0, 1.0), 2.0) * clamp(shaped, 0.0, 1.0);
            vec3  shaded = base + glint * 0.5 * typeTint;

            // crest feature: white foam for water, glow for lava, density for fog
            float crestMask = smoothstep(uFoamThreshold, uFoamThreshold + 0.15, shaped) * gradMag;

            if (foamy)
            {
                shaded = mix(shaded, vec3(1.0), crestMask * 0.85);
            }

            if (uFlowType == 1) // lava glow
            {
                vec3 glow = vec3(1.0, 0.35, 0.05) * (0.6 + 0.4 * sin(time * 2.0 + normHeight * 6.0));
                shaded = mix(shaded, glow, crestMask * 0.9);
            }
            else if (uFlowType == 2) // fog: modulate density/alpha instead of color
            {
                outAlpha *= mix(0.55, 1.0, 0.5 + 0.5 * shaped);
            }
            // water (uFlowType == 0) falls through untouched -- previously
            // it accidentally got fog's alpha modulation too

            outColor = shaded * uColor.rgb;
        }
    }

    if (uSelected)
    {
        outColor += selectionSparkle(vTex, uTimeUnscaled * 150.0);
    }

    FragColor = vec4(outColor, clamp(outAlpha * flowSample.a, 0.0, 1.0));
}
"""