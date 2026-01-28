var DigitizeDemo = (() => {
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined") return require.apply(this, arguments);
    throw Error('Dynamic require of "' + x + '" is not supported');
  });

  // src/digitize.ts
  var SIMPLE_REP_PATTERN = String.raw`(?:time|occurence|instance)s?`;
  var COMPLEX_REP_PATTERN = String.raw`(?:time|occurence|instance|attempt|try|tries)s?(?: (?:at|of))?(?: (?:which|when))?`;
  var PI_DECIMAL = "3.14159265358979323846264338327950288419716939937510";
  var SENTINEL = /* @__PURE__ */ Symbol("sentinel");
  function nonSentinels(p) {
    const out = {};
    for (const [k, v] of Object.entries(p)) {
      if (v !== SENTINEL) out[k] = v;
    }
    return out;
  }
  function replaceParams(base, patch) {
    return { ...base, ...patch };
  }
  function powBigInt(a, e) {
    if (e < 0n) throw new Error("negative exponent");
    let r = 1n, b = a, k = e;
    while (k > 0n) {
      if (k & 1n) r *= b;
      b *= b;
      k >>= 1n;
    }
    return r;
  }
  function fpMul(a, b, scale) {
    return a * b / scale;
  }
  function fpDiv(a, b, scale) {
    if (b === 0n) throw new Error("division by zero");
    return a * scale / b;
  }
  function fpPowInt(x, e, scale) {
    if (e < 0n) throw new Error("negative exponent");
    let r = scale;
    let b = x;
    let k = e;
    while (k > 0n) {
      if (k & 1n) r = fpMul(r, b, scale);
      b = fpMul(b, b, scale);
      k >>= 1n;
    }
    return r;
  }
  function roundHalfUp(numer, denom) {
    if (denom === 0n) throw new Error("division by zero");
    if (numer === 0n) return 0n;
    const neg = numer < 0n;
    let nn = neg ? -numer : numer;
    const dd = denom < 0n ? -denom : denom;
    let q = nn / dd;
    const r = nn % dd;
    if (2n * r >= dd) q += 1n;
    return neg ? -q : q;
  }
  function decPowRationalExpExactDecimal(base, exp, res) {
    if (base.n <= 0n) throw new Error("decimal approx only for positive base");
    if (res < 0) throw new Error("bad res");
    const p0 = exp.n;
    const q = exp.d;
    if (q <= 0n) throw new Error("bad exponent");
    const negExp = p0 < 0n;
    const p = negExp ? -p0 : p0;
    const prec = Math.max(50, res + 25);
    const SCALE = powBigInt(10n, BigInt(prec));
    const OUTSCALE = res === 0 ? 1n : powBigInt(10n, BigInt(res));
    const B = base.n * SCALE / base.d;
    const qMinus1 = q - 1n;
    let y = B < SCALE ? B : B / q;
    if (y <= 0n) y = 1n;
    for (let iter = 0; iter < 80; iter++) {
      const yPrev = y;
      const yPow = qMinus1 === 0n ? SCALE : fpPowInt(y, qMinus1, SCALE);
      if (yPow === 0n) break;
      const term = fpDiv(B, yPow, SCALE);
      const num = qMinus1 * y + term;
      y = num / q;
      if (y === yPrev) break;
    }
    let val = p === 0n ? SCALE : fpPowInt(y, p, SCALE);
    if (negExp) {
      val = fpDiv(SCALE, val, SCALE);
    }
    const numer = val * OUTSCALE;
    const R = roundHalfUp(numer, SCALE);
    return new Fraction(R, OUTSCALE);
  }
  function bgcd(a, b) {
    a = a < 0n ? -a : a;
    b = b < 0n ? -b : b;
    while (b !== 0n) {
      const t = a % b;
      a = b;
      b = t;
    }
    return a;
  }
  var Fraction = class _Fraction {
    n;
    d;
    constructor(n, d = 1n) {
      if (d === 0n) throw new Error("division by zero");
      const sign = d < 0n ? -1n : 1n;
      n *= sign;
      d *= sign;
      const g = bgcd(n, d);
      this.n = n / g;
      this.d = d / g;
    }
    static fromInt(x) {
      return new _Fraction(BigInt(x), 1n);
    }
    static fromDecimalString(lit) {
      const s = lit.trim();
      const neg = s.startsWith("-");
      const t = neg ? s.slice(1) : s;
      const [a, b = ""] = t.split(".");
      const num = BigInt(a + b);
      const den = 10n ** BigInt(b.length);
      return new _Fraction(neg ? -num : num, den === 0n ? 1n : den);
    }
    add(o) {
      return new _Fraction(this.n * o.d + o.n * this.d, this.d * o.d);
    }
    sub(o) {
      return new _Fraction(this.n * o.d - o.n * this.d, this.d * o.d);
    }
    mul(o) {
      return new _Fraction(this.n * o.n, this.d * o.d);
    }
    div(o) {
      return new _Fraction(this.n * o.d, this.d * o.n);
    }
    neg() {
      return new _Fraction(-this.n, this.d);
    }
    powInt(exp) {
      if (exp === 0n) return new _Fraction(1n, 1n);
      if (exp < 0n) return new _Fraction(this.d, this.n).powInt(-exp);
      return new _Fraction(this.n ** exp, this.d ** exp);
    }
  };
  function isTerminatingDen(d) {
    d = d < 0n ? -d : d;
    if (d === 0n) return false;
    while (d % 2n === 0n) d /= 2n;
    while (d % 5n === 0n) d /= 5n;
    return d === 1n;
  }
  function fractionToExactString(x) {
    const n = x.n;
    const d = x.d;
    if (!isTerminatingDen(d)) return `${n}/${d}`;
    const sign = n < 0n ? "-" : "";
    let nn = n < 0n ? -n : n;
    let dd = d;
    let k = 0n;
    while (dd % 2n === 0n) {
      dd /= 2n;
      k++;
    }
    while (dd % 5n === 0n) {
      dd /= 5n;
      k++;
    }
    if (k === 0n) return sign + nn.toString();
    const scale = 10n ** k;
    const scaled = nn * scale / d;
    let s = scaled.toString();
    const kk = Number(k);
    if (s.length <= kk) s = "0".repeat(kk - s.length + 1) + s;
    const out = (s.slice(0, s.length - kk) + "." + s.slice(s.length - kk)).replace(/\.?0+$/, "");
    return sign + out;
  }
  function tokenizeMath(expr) {
    const s = expr.trim();
    const toks = [];
    let i = 0;
    const isSpace = (c) => /\s/.test(c);
    while (i < s.length) {
      const c = s[i];
      if (isSpace(c)) {
        i++;
        continue;
      }
      if (c === "(") {
        toks.push({ t: "lp" });
        i++;
        continue;
      }
      if (c === ")") {
        toks.push({ t: "rp" });
        i++;
        continue;
      }
      if (c === "*" && s[i + 1] === "*") {
        toks.push({ t: "op", v: "**" });
        i += 2;
        continue;
      }
      if (c === "+" || c === "-" || c === "*" || c === "/") {
        toks.push({ t: "op", v: c });
        i++;
        continue;
      }
      const m = s.slice(i).match(/^\d+(?:\.\d+)?(?:\/\d+(?:\.\d+)?)?/);
      if (m) {
        toks.push({ t: "num", v: m[0] });
        i += m[0].length;
        continue;
      }
      throw new Error(`Bad token near: ${s.slice(i, i + 20)}`);
    }
    return toks;
  }
  function hasRealMath(expr) {
    try {
      const e = expr.replace(/\bpi\b/gi, "3.14159");
      const toks = tokenizeMath(e);
      const nums = toks.filter((t) => t.t === "num").length;
      const ops = toks.filter((t) => t.t === "op").length;
      return nums >= 2 && ops >= 1;
    } catch {
      return false;
    }
  }
  var PREC = {
    "**": { p: 4, a: "R" },
    "*": { p: 3, a: "L" },
    "/": { p: 3, a: "L" },
    "+": { p: 2, a: "L" },
    "-": { p: 2, a: "L" }
  };
  function toRPN(toks) {
    const out = [];
    const stack = [];
    const norm = [];
    for (let i = 0; i < toks.length; i++) {
      const t = toks[i];
      if (t.t === "op" && (t.v === "+" || t.v === "-")) {
        const prev = norm[norm.length - 1];
        const isUnary = !prev || prev.t === "op" || prev.t === "lp";
        if (isUnary) {
          norm.push({ t: "num", v: "0" });
        }
      }
      norm.push(t);
    }
    for (const t of norm) {
      if (t.t === "num") out.push(t);
      else if (t.t === "op") {
        while (stack.length) {
          const top = stack[stack.length - 1];
          if (top.t !== "op") break;
          const a = PREC[t.v];
          const b = PREC[top.v];
          if (a.a === "L" && a.p <= b.p || a.a === "R" && a.p < b.p) {
            out.push(stack.pop());
          } else break;
        }
        stack.push(t);
      } else if (t.t === "lp") stack.push(t);
      else if (t.t === "rp") {
        while (stack.length && stack[stack.length - 1].t !== "lp") out.push(stack.pop());
        if (!stack.length) throw new Error("mismatched parens");
        stack.pop();
      }
    }
    while (stack.length) {
      const t = stack.pop();
      if (t.t === "lp" || t.t === "rp") throw new Error("mismatched parens");
      out.push(t);
    }
    return out;
  }
  function parseNumToFraction(v) {
    if (v.includes("/")) {
      const [a, b] = v.split("/", 2);
      const fa = a.includes(".") ? Fraction.fromDecimalString(a) : new Fraction(BigInt(a), 1n);
      const fb = b.includes(".") ? Fraction.fromDecimalString(b) : new Fraction(BigInt(b), 1n);
      return fa.div(fb);
    }
    if (v.includes(".")) return Fraction.fromDecimalString(v);
    return new Fraction(BigInt(v), 1n);
  }
  function intNthRootExact(n, k) {
    if (n < 0n) return null;
    if (k === 1n) return n;
    if (n === 0n || n === 1n) return n;
    if (k === 2n) {
      let x = n;
      let y = (x + 1n) / 2n;
      while (y < x) {
        x = y;
        y = (x + n / x) / 2n;
      }
      return x * x === n ? x : null;
    }
    let lo = 0n, hi = 1n;
    const pow = (a, e) => a ** e;
    while (pow(hi, k) < n) hi *= 2n;
    while (lo + 1n < hi) {
      const mid = (lo + hi) / 2n;
      const p = pow(mid, k);
      if (p === n) return mid;
      if (p < n) lo = mid;
      else hi = mid;
    }
    return null;
  }
  function powFractionExact(base, exp) {
    if (exp.n === 0n) return new Fraction(1n, 1n);
    if (exp.d === 1n) return base.powInt(exp.n);
    const p = exp.n;
    const q = exp.d;
    if (base.n === 0n) return p > 0n ? new Fraction(0n, 1n) : null;
    if (base.n < 0n) return null;
    const a = base.n;
    const b = base.d;
    const ra = intNthRootExact(a, q);
    const rb = intNthRootExact(b, q);
    if (ra === null || rb === null) return null;
    const rooted = new Fraction(ra, rb);
    if (p >= 0n) return rooted.powInt(p);
    return new Fraction(1n, 1n).div(rooted.powInt(-p));
  }
  function piForRes(res) {
    const guard = 25;
    const digits = Math.min(PI_DECIMAL.length, (res ?? 3) + guard + 2);
    return PI_DECIMAL.slice(0, digits);
  }
  function safeEvalExpr(expr, evalFractions = false, res = 3) {
    if (evalFractions) {
      const piLit = piForRes(res);
      expr = expr.replace(/\bpi\b/gi, piLit);
    }
    const toks = tokenizeMath(expr);
    const rpn = toRPN(toks);
    const st = [];
    for (const t of rpn) {
      if (t.t === "num") {
        st.push(parseNumToFraction(t.v));
        continue;
      }
      if (t.t === "op") {
        const b = st.pop();
        const a = st.pop();
        if (!a || !b) throw new Error("bad expr");
        if (t.v === "+") st.push(a.add(b));
        else if (t.v === "-") st.push(a.sub(b));
        else if (t.v === "*") st.push(a.mul(b));
        else if (t.v === "/") st.push(a.div(b));
        else if (t.v === "**") {
          if (b.d === 1n) {
            st.push(a.powInt(b.n));
          } else {
            const got = powFractionExact(a, b);
            if (got) {
              st.push(got);
            } else {
              if (!evalFractions) throw new Error("non-integer exponent");
              const rr = typeof res === "number" ? res : 3;
              st.push(decPowRationalExpExactDecimal(a, b, rr));
            }
          }
        }
        continue;
      }
      throw new Error("bad rpn");
    }
    if (st.length !== 1) throw new Error("bad expr");
    return st[0];
  }
  function simpleEval(s, {
    power = "**",
    mult = "*",
    div = "/",
    evalFractions = false,
    res = 3,
    maxDecimalDigits = 2e3
  } = {}) {
    let t = s.split(power).join("**").split(mult).join("*").split(div).join("/");
    const exprRx = /(?<!\S)[0-9().\s+\-*/pi]+(?!\S)/gi;
    const matches = [...t.matchAll(exprRx)];
    for (let mi = matches.length - 1; mi >= 0; mi--) {
      const m = matches[mi];
      const expr = m[0];
      if (!hasRealMath(expr)) continue;
      let val;
      try {
        val = safeEvalExpr(expr.trim(), evalFractions, res);
      } catch {
        continue;
      }
      let out = fractionToExactString(val);
      if (evalFractions && out.includes("/") && typeof res === "number" && res <= maxDecimalDigits) {
        const n = val.n;
        const d = val.d;
        const neg = n < 0n !== d < 0n;
        let nn = n < 0n ? -n : n;
        let dd = d < 0n ? -d : d;
        const q = nn / dd;
        let r = nn % dd;
        if (res === 0) {
          if (2n * r >= dd) {
            const qq = q + 1n;
            out = (neg && qq !== 0n ? "-" : "") + qq.toString();
          } else {
            out = (neg && q !== 0n ? "-" : "") + q.toString();
          }
        } else {
          const scale = 10n ** BigInt(res);
          let scaled = r * scale / dd;
          const rem = r * scale % dd;
          if (2n * rem >= dd) {
            scaled += 1n;
            if (scaled === scale) {
              scaled = 0n;
              const qq = q + 1n;
              const frac2 = scaled.toString().padStart(res, "0");
              out = `${qq}.${frac2}`.replace(/\.?0+$/, "");
              if (neg && out !== "0") out = "-" + out;
              t = t.slice(0, m.index) + out + t.slice(m.index + expr.length);
              continue;
            }
          }
          const frac = scaled.toString().padStart(res, "0");
          out = `${q}.${frac}`.replace(/\.?0+$/, "");
          if (neg && out !== "0") out = "-" + out;
        }
      }
      t = t.slice(0, m.index) + out + t.slice(m.index + expr.length);
    }
    t = t.split("**").join(power).split("*").join(mult).split("/").join(div);
    return t;
  }
  var defaultParams = {
    description: "Tries to respect human language. Pretty and semi-parseable",
    config: "default",
    useCommas: false,
    fmt: "%n",
    replaceMultipliers: true,
    fmtMultipliers: null,
    supportOrdinals: true,
    fmtOrdinal: null,
    repSignifiers: COMPLEX_REP_PATTERN,
    supportReps: true,
    fmtRep: null,
    fmtNthTime: null,
    repFmt: "time",
    repFmtPlural: true,
    attemptToDifferentiateSeconds: true,
    literalFmt: false,
    supportRoman: false,
    parseSigns: true,
    power: "**",
    mult: "*",
    div: "/",
    combineAdd: true,
    res: null,
    doSimpleEvals: true,
    doFractionEvals: true,
    breaks: []
  };
  var modes = {
    default: defaultParams,
    nomath: replaceParams(defaultParams, {
      combineAdd: false,
      doSimpleEvals: false,
      doFractionEvals: false
    }),
    simplemath: replaceParams(defaultParams, {
      combineAdd: true,
      doSimpleEvals: true,
      res: null,
      doFractionEvals: false
    }),
    token: replaceParams(defaultParams, {
      description: "ugly but parseable",
      config: "token",
      fmt: "[NUM=%n,OG=%i]",
      fmtMultipliers: "[NUM=%n,MULT=%m,OG=%i]",
      fmtOrdinal: "[NUM=%n,ORD=%o,OG=%i]",
      fmtRep: "[NUM=%n,REP=%r,OG=%i]",
      fmtNthTime: "[NUM=%n,ORD=%o,REP=%r,OG=%i]"
    }),
    strip: replaceParams(defaultParams, {
      description: "simplifies the string a lot but very lossy of n-th n-th times, etc",
      config: "strip",
      repSignifiers: COMPLEX_REP_PATTERN,
      fmtOrdinal: "%n",
      fmtRep: "%n",
      fmtNthTime: "%n"
    }),
    nums: replaceParams(defaultParams, {
      description: "do not even look for once, n times, etc.",
      config: "num",
      supportReps: false,
      attemptToDifferentiateSeconds: false
    }),
    norm: replaceParams(defaultParams, {
      description: "Not grammatically correct but more parseable. e.g. 1-th, 2-th, 3-th time, etc",
      config: "norm",
      fmtOrdinal: "%n-th",
      fmtRep: "%n-th time"
    })
  };
  function ordinalSuffix(n) {
    const nAbs = Math.abs(n);
    const lastTwo = nAbs % 100;
    if (11 <= lastTwo && lastTwo <= 13) return "th";
    const last = nAbs % 10;
    if (last === 1) return "st";
    if (last === 2) return "nd";
    if (last === 3) return "rd";
    return "th";
  }
  var ROMAN_PATTERN = /^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$/i;
  function romanToInt(s) {
    const roman = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1e3 };
    const t = s.toUpperCase();
    let num = 0;
    for (let i = 0; i < t.length - 1; i++) {
      if (roman[t[i]] < roman[t[i + 1]]) num -= roman[t[i]];
      else num += roman[t[i]];
    }
    num += roman[t[t.length - 1]];
    return num;
  }
  function digitize(input, opts = {}) {
    const cfgRaw = opts.config ?? "default";
    const defaults = typeof cfgRaw === "object" && cfgRaw && "fmt" in cfgRaw ? cfgRaw : modes[cfgRaw] ?? modes.default;
    const params = {
      description: SENTINEL,
      config: SENTINEL,
      useCommas: opts.useCommas ?? SENTINEL,
      fmt: opts.fmt ?? SENTINEL,
      replaceMultipliers: opts.replaceMultipliers ?? SENTINEL,
      fmtMultipliers: opts.fmtMultipliers ?? SENTINEL,
      supportOrdinals: opts.supportOrdinals ?? SENTINEL,
      fmtOrdinal: opts.fmtOrdinal ?? SENTINEL,
      repSignifiers: opts.repSignifiers ?? SENTINEL,
      supportReps: opts.supportReps ?? SENTINEL,
      fmtRep: opts.fmtRep ?? SENTINEL,
      fmtNthTime: opts.fmtNthTime ?? SENTINEL,
      repFmt: opts.repFmt ?? SENTINEL,
      repFmtPlural: opts.repFmtPlural ?? SENTINEL,
      attemptToDifferentiateSeconds: opts.attemptToDifferentiateSeconds ?? SENTINEL,
      literalFmt: opts.literalFmt ?? SENTINEL,
      supportRoman: opts.supportRoman ?? SENTINEL,
      parseSigns: opts.parseSigns ?? SENTINEL,
      power: opts.power ?? SENTINEL,
      mult: opts.mult ?? SENTINEL,
      div: opts.div ?? SENTINEL,
      combineAdd: opts.combineAdd ?? SENTINEL,
      res: opts.res ?? SENTINEL,
      doSimpleEvals: opts.doSimpleEvals ?? SENTINEL,
      doFractionEvals: opts.doFractionEvals ?? SENTINEL,
      breaks: opts.breaks ?? SENTINEL
    };
    const config = { ...nonSentinels(defaults), ...nonSentinels(params) };
    const power = config.power;
    const mult = config.mult;
    const div = config.div;
    let s = input;
    let breaks = [];
    if (typeof config.breaks === "string") breaks = [config.breaks];
    else if (config.breaks && Symbol.iterator in Object(config.breaks)) breaks = Array.from(config.breaks);
    if (breaks.length) {
      const esc = (x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const pattern = new RegExp("(" + breaks.map(esc).join("|") + ")", "g");
      const parts = s.split(pattern);
      const chunks = parts.filter((_, i) => i % 2 === 0);
      const seps = parts.filter((_, i) => i % 2 === 1);
      const processed = chunks.map((c) => digitize(c, { ...opts, config }));
      let out2 = "";
      for (let i = 0; i < processed.length; i++) {
        out2 += processed[i];
        if (i < seps.length) out2 += seps[i];
      }
      return out2;
    }
    let fmt = config.fmt;
    let fmtMultipliers = config.fmtMultipliers;
    let fmtOrdinal = config.fmtOrdinal;
    let fmtRep = config.fmtRep;
    let fmtNthTime = config.fmtNthTime;
    const literalFmt = config.literalFmt;
    if (!literalFmt) fmt = fmt.replace(/\d+/g, "%n");
    if (fmtMultipliers == null) fmtMultipliers = fmt;
    if (!literalFmt) fmtMultipliers = fmtMultipliers.replace(/\d+/g, "%n");
    if (fmtOrdinal == null) fmtOrdinal = fmt.replace("%n", "%n%o");
    if (!literalFmt) fmtOrdinal = fmtOrdinal.replace(/\d+/g, "%n");
    if (fmtRep == null) {
      if (config.repFmt === "x") fmtRep = fmt.replace("%n", "%nx");
      else {
        const repFmt = config.repFmt;
        fmtRep = fmt.replace("%n", config.repFmtPlural ? `%n ${repFmt}%s` : `%n ${repFmt}`);
      }
    }
    if (!literalFmt) fmtRep = fmtRep.replace(/\d+/g, "%n");
    if (fmtNthTime == null) {
      fmtNthTime = (fmtOrdinal ?? fmt).replace(/%n(%o)?/g, `%n$1 ${config.repFmt}`);
    }
    if (!literalFmt) fmtNthTime = fmtNthTime.replace(/\d+/g, "%n");
    if (config.parseSigns && !literalFmt) {
      const addP = (x) => x.replace(/%n/g, "%p%n");
      fmt = addP(fmt);
      fmtMultipliers = addP(fmtMultipliers);
      fmtOrdinal = addP(fmtOrdinal);
      fmtRep = addP(fmtRep);
      fmtNthTime = addP(fmtNthTime);
    }
    const SEC_SENTINEL = "__DIGITIZE_ECOND_UNIT__";
    const REPEAT_PREFIX = "__repeat__";
    const repeatMap = /* @__PURE__ */ new Map();
    if (config.supportReps && config.repSignifiers) {
      let repRx;
      const repSignifiers = config.repSignifiers;
      if (typeof repSignifiers === "string") repRx = new RegExp(String.raw`(?i)\b(${repSignifiers})\b`.replace("(?i)", ""), "gi");
      else if (repSignifiers instanceof RegExp) repRx = repSignifiers;
      else {
        const pats = repSignifiers.map((p) => p instanceof RegExp ? p.source : p);
        repRx = new RegExp(String.raw`\b(` + pats.join("|") + String.raw`)\b`, "gi");
      }
      s = s.replace(repRx, (m) => {
        const key = `${REPEAT_PREFIX}${repeatMap.size}_`;
        repeatMap.set(key, m);
        return key;
      });
    }
    if (config.attemptToDifferentiateSeconds) {
      s = s.replace(/\b(a|one|per|each)\s+(second)\b/gi, (_m, g1) => `${g1} ${SEC_SENTINEL}`);
      s = s.replace(/\b(the)\s+(second)\s+(after|before|between|when)\b/gi, (_m, g1, _g2, g3) => `${g1} ${SEC_SENTINEL} ${g3}`);
    }
    if (config.replaceMultipliers) {
      const suffixMultipliers = { k: 1e3, K: 1e3, M: 1e6, G: 1e9, T: 1e12, P: 1e15 };
      s = s.replace(/(?<![A-Za-z0-9])(\d+(?:\.\d+)?)([kKMGTP])(?=[^A-Za-z0-9]|$)/g, (_m, n1, suf) => {
        const value = Math.trunc(parseFloat(n1) * suffixMultipliers[suf]);
        const nFmt = config.useCommas ? value.toLocaleString("en-US") : String(value);
        return fmtMultipliers.replace("%n", nFmt).replace("%m", suf).replace("%i", `${n1}${suf}`);
      });
    }
    const known = [
      [/dozens?/gi, 12],
      [/baker's dozens?/gi, 13],
      [/pairs?(?: of)/gi, 2]
    ];
    for (const [rx, v] of known) {
      s = s.replace(new RegExp(`\\ba ${rx.source}\\b`, "gi"), String(v));
      s = s.replace(new RegExp(`\\b${rx.source}\\b`, "gi"), `set of ${v}`);
    }
    const n019 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
    const digitWordToDigit = { o: "0", oh: "0", zero: "0", one: "1", two: "2", three: "3", four: "4", five: "5", six: "6", seven: "7", eight: "8", nine: "9" };
    const tensWords = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    const wordToNum = {};
    n019.forEach((w, i) => wordToNum[w] = i);
    tensWords.forEach((w, i) => wordToNum[w] = (i + 2) * 10);
    const magnitudeValue = {
      thousand: 1e3,
      million: 1e6,
      billion: 1e9,
      trillion: 1e12,
      quadrillion: 1e15
    };
    const ordinalWordToNum = {};
    const ordinalMagnitudeExact = {};
    const fractionDenWord = {
      half: 2,
      halves: 2,
      third: 3,
      thirds: 3,
      hundredth: 100,
      hundredths: 100,
      thousandth: 1e3,
      thousandths: 1e3,
      millionth: 1e6,
      millionths: 1e6,
      billionth: 1e9,
      billionths: 1e9,
      trillionth: 1e12,
      trillionths: 1e12,
      quadrillionth: 1e15,
      quadrillionths: 1e15
    };
    if (config.supportOrdinals) {
      Object.assign(ordinalWordToNum, {
        first: 1,
        second: 2,
        third: 3,
        fourth: 4,
        fifth: 5,
        sixth: 6,
        seventh: 7,
        eighth: 8,
        ninth: 9,
        tenth: 10,
        eleventh: 11,
        twelfth: 12,
        thirteenth: 13,
        fourteenth: 14,
        fifteenth: 15,
        sixteenth: 16,
        seventeenth: 17,
        eighteenth: 18,
        nineteenth: 19,
        fifthe: 5,
        twentieth: 20,
        thirtieth: 30,
        fortieth: 40,
        fiftieth: 50,
        sixtieth: 60,
        seventieth: 70,
        eightieth: 80,
        ninetieth: 90
      });
      Object.assign(ordinalMagnitudeExact, {
        hundredth: 100,
        thousandth: 1e3,
        millionth: 1e6,
        billionth: 1e9,
        trillionth: 1e12,
        quadrillionth: 1e15
      });
    }
    const repeatWordToNum = { once: 1, twice: 2, thrice: 3 };
    const isRepeatTail = (w) => new RegExp(`^${REPEAT_PREFIX}\\d+_$`, "i").test(w);
    function isNumericAtom(tok) {
      const t = tok.toLowerCase();
      if (t === "point") return true;
      if (t === "oh" || t === "o") return true;
      if (t === "pi") return true;
      if (t in fractionDenWord) return true;
      if (config.supportReps && t in repeatWordToNum) return true;
      if (config.supportOrdinals && /^\d+(st|nd|rd|th)$/i.test(t)) return true;
      if (/^\d+$/.test(t) || t in wordToNum || t === "hundred" || t in magnitudeValue) return true;
      if (config.supportOrdinals && (t in ordinalWordToNum || t in ordinalMagnitudeExact)) return true;
      if (config.supportReps && new RegExp(`^${REPEAT_PREFIX}\\d+_$`, "i").test(tok)) return true;
      if (config.supportRoman && ROMAN_PATTERN.test(tok)) return true;
      return false;
    }
    function allowsAndAfter(prevNorm) {
      if (!prevNorm) return false;
      const p = prevNorm.toLowerCase();
      return p === "hundred" || p in magnitudeValue;
    }
    ;
    function parseNumber(normWords) {
      if (normWords.length && normWords[normWords.length - 1] === "and") return null;
      if (config.supportReps && normWords.length) {
        const w0 = normWords[0];
        if (w0 in repeatWordToNum) {
          if (normWords.length === 1) return { kind: "int", n: repeatWordToNum[w0], isOrd: false, isTime: true };
          if (isRepeatTail(normWords[normWords.length - 1])) return { kind: "int", n: repeatWordToNum[w0], isOrd: false, isTime: true };
        }
      }
      let isTime = false;
      let core = normWords;
      if (config.supportReps && core.length && isRepeatTail(core[core.length - 1])) {
        isTime = true;
        core = core.slice(0, -1);
      }
      if (!core.length) return null;
      if (core.length === 1 && core[0] === "pi") {
        return { kind: "pi", coef: 1, isTime: false };
      }
      if (core.length === 2 && core[1] === "pi") {
        const k = core[0];
        if (/^\d+$/.test(k)) return { kind: "pi", coef: parseInt(k, 10), isTime: false };
        if (k in wordToNum) return { kind: "pi", coef: wordToNum[k], isTime: false };
      }
      const denomTok = core[core.length - 1];
      let denomVal = null;
      let denomIsPlural = false;
      if (!isTime) {
        if (denomTok in fractionDenWord) {
          denomVal = fractionDenWord[denomTok];
          denomIsPlural = denomTok.endsWith("s") || denomTok.endsWith("ves");
        }
        if (denomVal == null) {
          const mo = denomTok.match(/^(\d+)(st|nd|rd|th)(s)?$/i);
          if (mo && mo[2].toLowerCase() === "th") {
            denomVal = parseInt(mo[1], 10);
            denomIsPlural = !!mo[3];
          }
        }
        if (denomVal != null) {
          const numerWords = core.slice(0, -1);
          if (!numerWords.length) {
            if (denomTok === "half" || denomTok === "halves") return { kind: "frac", numer: "1", denom: denomVal, isTime: false };
            denomVal = null;
          }
          if (denomVal != null) {
            let numerStr = null;
            let numerIsOneish = false;
            if (numerWords.length === 1 && numerWords[0] === "a") {
              numerStr = "1";
              numerIsOneish = true;
            } else {
              const parseNumericString = (words) => {
                if (words.includes("point")) {
                  const p = words.indexOf("point");
                  const left = words.slice(0, p);
                  const right = words.slice(p + 1);
                  if (!right.length) return null;
                  const digs = [];
                  for (const w of right) {
                    if (w === "and") return null;
                    if (/^\d+$/.test(w)) digs.push(w);
                    else if (w in digitWordToDigit) digs.push(digitWordToDigit[w]);
                    else return null;
                  }
                  const frac = digs.join("");
                  if (!frac) return null;
                  let ip = 0;
                  if (left.length) {
                    let total3 = 0, current3 = 0;
                    let saw3 = false;
                    for (const w of left) {
                      if (w === "and") continue;
                      if (w === "oh" || w === "o") {
                        saw3 = true;
                        continue;
                      }
                      if (/^\d+$/.test(w)) {
                        current3 += parseInt(w, 10);
                        saw3 = true;
                        continue;
                      }
                      if (w in wordToNum) {
                        current3 += wordToNum[w];
                        saw3 = true;
                        continue;
                      }
                      if (w === "hundred") {
                        if (!saw3) return null;
                        current3 *= 100;
                        continue;
                      }
                      if (w in magnitudeValue) {
                        if (!saw3) return null;
                        total3 += current3 * magnitudeValue[w];
                        current3 = 0;
                        continue;
                      }
                      return null;
                    }
                    if (!saw3) return null;
                    ip = total3 + current3;
                  }
                  return `${ip}.${frac}`;
                }
                let total2 = 0, current2 = 0;
                let saw2 = false;
                for (const w of words) {
                  if (w === "and") continue;
                  if (w === "oh" || w === "o") {
                    saw2 = true;
                    continue;
                  }
                  if (/^\d+$/.test(w)) {
                    current2 += parseInt(w, 10);
                    saw2 = true;
                    continue;
                  }
                  if (w in wordToNum) {
                    current2 += wordToNum[w];
                    saw2 = true;
                    continue;
                  }
                  if (w === "hundred") {
                    if (!saw2) return null;
                    current2 *= 100;
                    continue;
                  }
                  if (w in magnitudeValue) {
                    if (!saw2) return null;
                    total2 += current2 * magnitudeValue[w];
                    current2 = 0;
                    continue;
                  }
                  return null;
                }
                if (!saw2) return null;
                return String(total2 + current2);
              };
              numerStr = parseNumericString(numerWords);
              if (numerStr == null) return null;
              numerIsOneish = numerStr === "1" || numerStr === "1.0";
            }
            if (!(denomIsPlural || numerIsOneish || numerStr != null)) return null;
            return { kind: "frac", numer: numerStr, denom: denomVal, isTime: false };
          }
        }
      }
      if (core.includes("point")) {
        const p = core.indexOf("point");
        const left = core.slice(0, p);
        const right = core.slice(p + 1);
        if (!right.length) return null;
        const digits = [];
        for (const w of right) {
          if (w === "and") return null;
          if (/^\d+$/.test(w)) {
            digits.push(w);
            continue;
          }
          if (w in digitWordToDigit) {
            digits.push(digitWordToDigit[w]);
            continue;
          }
          return null;
        }
        const fracDigits = digits.join("");
        if (!fracDigits) return null;
        let intPart = 0;
        if (left.length) {
          let total2 = 0, current2 = 0, saw2 = false;
          for (const w of left) {
            if (w === "and") continue;
            if (w === "oh" || w === "o") {
              saw2 = true;
              continue;
            }
            if (config.supportOrdinals) {
              if (/^\d+(st|nd|rd|th)$/i.test(w)) return null;
              if (w in ordinalWordToNum || w in ordinalMagnitudeExact) return null;
            }
            if (/^\d+$/.test(w)) {
              current2 += parseInt(w, 10);
              saw2 = true;
              continue;
            }
            if (w in wordToNum) {
              current2 += wordToNum[w];
              saw2 = true;
              continue;
            }
            if (w === "hundred") {
              if (!saw2) return null;
              current2 *= 100;
              continue;
            }
            if (w in magnitudeValue) {
              if (!saw2) return null;
              total2 += current2 * magnitudeValue[w];
              current2 = 0;
              continue;
            }
            return null;
          }
          if (!saw2) return null;
          intPart = total2 + current2;
        }
        return { kind: "dec", intPart, fracDigits };
      }
      let total = 0, current = 0;
      let saw = false;
      let isOrd = false;
      for (const w of core) {
        if (w === "and") continue;
        if (config.supportOrdinals) {
          const mo = w.match(/^(\d+)(st|nd|rd|th)$/i);
          if (mo) {
            current += parseInt(mo[1], 10);
            saw = true;
            isOrd = true;
            continue;
          }
        }
        if (/^\d+$/.test(w)) {
          current += parseInt(w, 10);
          saw = true;
          continue;
        }
        if (w in wordToNum) {
          current += wordToNum[w];
          saw = true;
          continue;
        }
        if (config.supportOrdinals && w in ordinalWordToNum) {
          current += ordinalWordToNum[w];
          saw = true;
          isOrd = true;
          continue;
        }
        if (w === "hundred") {
          if (!saw) return null;
          current *= 100;
          continue;
        }
        if (config.supportOrdinals && w in ordinalMagnitudeExact) {
          if (!saw) {
            current = ordinalMagnitudeExact[w];
            saw = true;
          } else current *= ordinalMagnitudeExact[w];
          isOrd = true;
          continue;
        }
        if (w in magnitudeValue) {
          if (!saw) return null;
          total += current * magnitudeValue[w];
          current = 0;
          continue;
        }
        if (config.supportRoman && ROMAN_PATTERN.test(w)) {
          current += romanToInt(w);
          saw = true;
          continue;
        }
        return null;
      }
      if (!saw) return null;
      return { kind: "int", n: total + current, isOrd, isTime };
    }
    const tokenRx = new RegExp(
      String.raw`${REPEAT_PREFIX}\d+_|\d+(?:st|nd|rd|th)|[A-Za-z]+|\d+|\s+|[^A-Za-z\d\s]+`,
      "gi"
    );
    const tokens = s.match(tokenRx) ?? [];
    const out = [];
    let raw = [];
    let norm = [];
    let pendingWs = "";
    const nextNonSpace = (j) => {
      let k = j + 1;
      while (k < tokens.length && /^\s+$/.test(tokens[k])) k++;
      return k < tokens.length ? tokens[k] : null;
    };
    const peekDecimalStart = (i) => {
      const nxt = nextNonSpace(i);
      if (!nxt || nxt.toLowerCase() !== "point") return false;
      let j = i + 1;
      while (j < tokens.length && /^\s+$/.test(tokens[j])) j++;
      if (j >= tokens.length || tokens[j].toLowerCase() !== "point") return false;
      let k = j + 1;
      while (k < tokens.length && /^\s+$/.test(tokens[k])) k++;
      if (k >= tokens.length) return false;
      const t2 = tokens[k].toLowerCase();
      return /^\d+$/.test(t2) || t2 in digitWordToDigit;
    };
    const commitPendingWsIntoPhrase = () => {
      if (pendingWs) {
        raw.push(pendingWs);
        pendingWs = "";
      }
    };
    const hyphenIsInternal = (prevNorm, nextTok) => {
      if (!prevNorm || !nextTok) return false;
      const p = prevNorm.toLowerCase();
      const nxt = nextTok.toLowerCase();
      if (tensWords.includes(p)) {
        if (/^\d+$/.test(nxt) || nxt in wordToNum) return true;
        if (config.supportOrdinals && nxt in ordinalWordToNum) return true;
      }
      return false;
    };
    const flushPhrase = () => {
      if (!norm.length) return;
      let hasConvertible = false;
      for (const w of norm) {
        if (w === "and") continue;
        if (w === "pi") {
          hasConvertible = true;
          break;
        }
        if (w in fractionDenWord) {
          hasConvertible = true;
          break;
        }
        if (/^\d+$/.test(w)) continue;
        if (w in wordToNum || w === "hundred" || w in magnitudeValue) {
          hasConvertible = true;
          break;
        }
        if (config.supportOrdinals && (w in ordinalWordToNum || w in ordinalMagnitudeExact || /^\d+(st|nd|rd|th)$/i.test(w))) {
          hasConvertible = true;
          break;
        }
        if (config.supportReps && new RegExp(`^${REPEAT_PREFIX}\\d+_$`, "i").test(w)) {
          hasConvertible = true;
          break;
        }
        if (config.supportReps && w in repeatWordToNum) {
          hasConvertible = true;
          break;
        }
        if (config.supportRoman && ROMAN_PATTERN.test(w)) {
          hasConvertible = true;
          break;
        }
      }
      if (!hasConvertible) {
        out.push(raw.join(""));
        raw = [];
        norm = [];
        if (pendingWs) {
          out.push(pendingWs);
          pendingWs = "";
        }
        return;
      }
      const parsed = parseNumber(norm);
      if (!parsed) {
        out.push(raw.join(""));
      } else {
        const iText = raw.join("");
        const useCommas = config.useCommas;
        if (parsed.kind === "pi") {
          const coef = parsed.coef;
          const expr = coef === 1 ? "pi" : `${coef}${mult}pi`;
          out.push(
            fmt.replace("%n", expr).replace("%s", "s").replace("%r", "x").replace("%i", iText)
          );
        } else if (parsed.kind === "frac") {
          const num = `${parsed.numer}/${parsed.denom}`;
          out.push(
            fmt.replace("%n", num).replace("%s", "s").replace("%r", "x").replace("%i", iText)
          );
        } else if (parsed.kind === "dec") {
          const intStr = useCommas ? parsed.intPart.toLocaleString("en-US") : String(parsed.intPart);
          const num = `${intStr}.${parsed.fracDigits}`;
          out.push(
            fmt.replace("%n", num).replace("%s", "s").replace("%r", "x").replace("%i", iText)
          );
        } else {
          const n = parsed.n;
          const num = useCommas ? n.toLocaleString("en-US") : String(n);
          const pluralS = Math.abs(n) !== 1 ? "s" : "";
          let r = "x";
          let i = iText;
          if (repeatMap.size) {
            const pat = new RegExp(`^(.*)?(${REPEAT_PREFIX}\\d+_)(.*)$`, "i");
            const m = iText.match(pat);
            if (m) {
              r = repeatMap.get(m[2]) ?? m[2];
              i = iText.replace(pat, `$1${r}$3`);
            }
          }
          if (parsed.isTime && config.supportReps) {
            if (parsed.isOrd && config.supportOrdinals) {
              const suf = ordinalSuffix(n);
              out.push(
                fmtNthTime.replace("%n", num).replace("%o", suf).replace("%s", pluralS).replace("%r", r).replace("%i", i)
              );
            } else {
              out.push(
                fmtRep.replace("%n", num).replace("%s", pluralS).replace("%r", r).replace("%i", i)
              );
            }
          } else {
            if (parsed.isOrd && config.supportOrdinals) {
              const suf = ordinalSuffix(n);
              out.push(
                fmtOrdinal.replace("%n", num).replace("%o", suf).replace("%s", pluralS).replace("%r", r).replace("%i", i)
              );
            } else {
              out.push(
                fmt.replace("%n", num).replace("%s", pluralS).replace("%r", r).replace("%i", i)
              );
            }
          }
        }
      }
      raw = [];
      norm = [];
      if (pendingWs) {
        out.push(pendingWs);
        pendingWs = "";
      }
    };
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      if (/^\s+$/.test(t)) {
        if (norm.length) pendingWs += t;
        else out.push(t);
        continue;
      }
      const isWordNumOrRepeat = new RegExp(`^${REPEAT_PREFIX}\\d+_$|^[A-Za-z]+$|^\\d+$|^\\d+(?:st|nd|rd|th)$`, "i").test(t);
      if (isWordNumOrRepeat) {
        const tl = t.toLowerCase();
        if (tl === "a") {
          const nxt = nextNonSpace(i);
          if (nxt && nxt.toLowerCase() in fractionDenWord) {
            if (norm.length) commitPendingWsIntoPhrase();
            raw.push(t);
            norm.push("a");
            continue;
          }
        }
        if (config.supportReps && new RegExp(`^${REPEAT_PREFIX}\\d+_$`, "i").test(t)) {
          const nxt = nextNonSpace(i);
          if (nxt && isNumericAtom(nxt)) {
            flushPhrase();
            out.push(repeatMap.get(t) ?? t);
            continue;
          }
          if (norm.length) {
            if (parseNumber(norm)) {
              commitPendingWsIntoPhrase();
              raw.push(t);
              norm.push(tl);
            } else {
              flushPhrase();
              out.push(repeatMap.get(t) ?? t);
            }
          } else {
            out.push(repeatMap.get(t) ?? t);
          }
          continue;
        }
        if (tl === "and") {
          if (norm.length) {
            const nxt = nextNonSpace(i);
            let prevNorm = null;
            for (let k = norm.length - 1; k >= 0; k--) {
              if (norm[k] !== "and") {
                prevNorm = norm[k];
                break;
              }
            }
            if (allowsAndAfter(prevNorm) && nxt && isNumericAtom(nxt)) {
              commitPendingWsIntoPhrase();
              raw.push(t);
              norm.push("and");
            } else {
              flushPhrase();
              out.push(t);
            }
          } else out.push(t);
          continue;
        }
        if ((tl === "oh" || tl === "o") && peekDecimalStart(i)) {
          if (norm.length) commitPendingWsIntoPhrase();
          raw.push(t);
          norm.push("zero");
          continue;
        }
        if (isNumericAtom(t)) {
          if (norm.length) commitPendingWsIntoPhrase();
          raw.push(t);
          norm.push(tl);
        } else {
          flushPhrase();
          out.push(t);
        }
        continue;
      }
      if (t === "-") {
        if (norm.length) {
          const nxt = nextNonSpace(i);
          let prevNorm = null;
          for (let k = norm.length - 1; k >= 0; k--) {
            if (norm[k] !== "and") {
              prevNorm = norm[k];
              break;
            }
          }
          if (hyphenIsInternal(prevNorm, nxt)) {
            commitPendingWsIntoPhrase();
            raw.push("-");
            continue;
          }
        }
        flushPhrase();
        out.push("-");
        continue;
      }
      flushPhrase();
      out.push(t);
    }
    flushPhrase();
    if (pendingWs) out.push(pendingWs);
    s = out.join("");
    if (config.attemptToDifferentiateSeconds) s = s.split(SEC_SENTINEL).join("second");
    if (repeatMap.size) {
      const repRx = new RegExp(`${REPEAT_PREFIX}\\d+_`, "gi");
      s = s.replace(repRx, (m) => repeatMap.get(m) ?? m);
    }
    s = s.replace(/%p/g, "");
    if (config.parseSigns) {
      const numRx = String.raw`(\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)`;
      s = s.replace(new RegExp(String.raw`\b(neg|negative|minus)\s+${numRx}\b`, "gi"), "-$2");
      s = s.replace(new RegExp(String.raw`\b(pos|positive|plus)\s+${numRx}\b`, "gi"), "+$2");
      s = s.replace(new RegExp(String.raw`\+\s+${numRx}\b`, "gi"), "+$1");
      s = s.replace(new RegExp(String.raw`-\s+${numRx}\b`, "gi"), "-$1");
    }
    s = s.replace(/\b([+-]?\d+(?:\.\d+)?)\s+(?:over|divided\s+by)\s+(\d+(?:\.\d+)?)\b/gi, `$1${div}$2`);
    s = s.replace(/\b([+-]?\d+(?:\.\d+)?)\s+(?:over|divided\s+by|out of|into|of)\s+(\d+(?:\.\d+)?)\b/gi, `$1${div}$2`);
    const numAtom = String.raw`(?:pi|[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)`;
    const numAtom2 = String.raw`(?:pi|[+-]?\d+(?:\.\d+)?(?:/\d+)?)`;
    const numAtomSci = String.raw`(?:pi|[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)`;
    s = s.replace(
      new RegExp(String.raw`\b(${numAtom})\s+(?:time|multiplied|timesed|occurence|instance|attempt|multiply|multiple|set)s?(?: (?:by|of))?\s+(${numAtom})\b`, "gi"),
      `$1${mult}$2`
    );
    s = s.replace(new RegExp(String.raw`\b(${numAtom})\s\+(${numAtom})\b`, "gi"), "$1+$2");
    s = s.replace(new RegExp(String.raw`\b(${numAtom})\s-(${numAtom})\b`, "gi"), "$1-$2");
    s = s.replace(
      new RegExp(String.raw`\b(${numAtom})\s(?:raised )?(?:to the power of|to the)\s(${numAtom})(?:rd|st|th)?(?: (?:power|exponent|degree))?\b`, "gi"),
      `$1${power}$2`
    );
    const powers = { squared: 2, cubed: 3 };
    for (const [k, v] of Object.entries(powers)) {
      s = s.replace(
        new RegExp(String.raw`\b(${numAtom})\s(?:raised )?(?:to the power of|to the)?\s(${k})(?: power)?\b`, "gi"),
        `$1${power}${v}`
      );
    }
    s = s.replace(new RegExp(String.raw`\b(square)\s+root(?:\s+of)?\s+(${numAtom2})\b`, "gi"), `$2${power}(1/2)`);
    s = s.replace(new RegExp(String.raw`\b(sqrt)\s+(?:\s+of)?\s+(${numAtom2})\b`, "gi"), `$2${power}(1/2)`);
    s = s.replace(new RegExp(String.raw`\b(cube)\s+root(?:\s+of)?\s+(${numAtom2})\b`, "gi"), `$2${power}(1/3)`);
    s = s.replace(new RegExp(String.raw`\b(?:the\s+)?(${numAtom2})(?:st|nd|rd|th)?\s+root(?:\s+of)?\s+(${numAtom2})\b`, "gi"), `$2${power}(1/$1)`);
    s = s.replace(new RegExp(String.raw`\b(${numAtomSci})\s*e\s*([+-]?\d+)\b`, "gi"), "$1e$2");
    s = s.replace(new RegExp(String.raw`\b([+-]?\d+(?:\.\d+)?)(?:e([+-]?\d+))\b`, "gi"), (_m, a, b) => `${a}${mult}10${power}${b}`);
    const tenPow = String.raw`10(?:\s*)?(?:\^|\*\*)(?:\s*)?(\(?[+-]?\d+\)?)`;
    s = s.replace(new RegExp(String.raw`\b(${numAtomSci})\s*(?:\*|x|×)\s*${tenPow}\b`, "gi"), (_m, a, b) => `${a}${mult}(10${power}${b})`);
    if (config.combineAdd !== false) {
      s = s.replace(/\b([+-]?\d+(?:\.\d+)?)\s+and\s+([+-]?\d+(?:\.\d+)?)\/(\d+)\b/gi, (_m, wholeS, numS, denS) => {
        try {
          const whole = wholeS.includes(".") ? Fraction.fromDecimalString(wholeS) : new Fraction(BigInt(wholeS), 1n);
          const num = numS.includes(".") ? Fraction.fromDecimalString(numS) : new Fraction(BigInt(numS), 1n);
          const den = new Fraction(BigInt(denS), 1n);
          const val = whole.add(num.div(den));
          if (config.res == null) {
            if (!isTerminatingDen(val.d)) return _m;
            return fractionToExactString(val);
          }
          return fractionToExactString(val);
        } catch {
          return _m;
        }
      });
    }
    if (config.doSimpleEvals) {
      s = simpleEval(s, {
        power,
        mult,
        div,
        evalFractions: !!config.doFractionEvals,
        res: typeof config.res === "number" ? config.res : 3
      });
    }
    if (config.doFractionEvals) {
      const rr = typeof config.res === "number" ? config.res : 3;
      const piLit = piForRes(rr);
      s = s.replace(/\bpi\b/gi, piLit);
    }
    return s;
  }
  var isNode = typeof process !== "undefined" && process.versions?.node && typeof __require !== "undefined" && typeof module !== "undefined";
  if (isNode && __require.main === module) {
    const argv = process.argv.slice(2);
    const text = argv.join(" ");
    if (!text) {
      let data = "";
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (c) => data += c);
      process.stdin.on("end", () => {
        const out = digitize(data.replace(/\n$/, ""));
        process.stdout.write(out + (out.endsWith("\n") ? "" : "\n"));
      });
    } else {
      const out = digitize(text);
      process.stdout.write(out + (out.endsWith("\n") ? "" : "\n"));
    }
  }

  // src/main.ts
  window.digitize = digitize;
})();
