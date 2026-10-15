
"use client";

import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

const FAMILIES = [
  "AUTOMOTIVE",
  "BABY CARE",
  "BEAUTY",
  "BEVERAGES",
  "BOOKS",
  "BREAD/BAKERY",
  "CELEBRATION",
  "CLEANING",
  "DAIRY",
  "DELI",
  "EGGS",
  "FROZEN FOODS",
  "GROCERY I",
  "GROCERY II",
  "HARDWARE",
  "HOME AND KITCHEN I",
  "HOME AND KITCHEN II",
  "HOME APPLIANCES",
  "HOME CARE",
  "LADIESWEAR",
  "LAWN AND GARDEN",
  "LINGERIE",
  "LIQUOR,WINE,BEER",
  "MAGAZINES",
  "MEATS",
  "PERSONAL CARE",
  "PET SUPPLIES",
  "PLAYERS AND ELECTRONICS",
  "POULTRY",
  "PREPARED FOODS",
  "PRODUCE",
  "SCHOOL AND OFFICE SUPPLIES",
  "SEAFOOD",
];

const HORIZONS = [7, 14, 30] as const;

type Horizon = (typeof HORIZONS)[number];

type ForecastDay = {
  date: string;
  store_nbr: number;
  family: string;
  onpromotion: number;
  predicted_sales: number;
  promotion_assumption: string;
};

type ForecastResponse = {
  forecast: ForecastDay[];
};

function formatDate(value: string, short = false) {
  const [year, month, day] = value.split("-").map(Number);
  const date = new Date(year, month - 1, day);

  return new Intl.DateTimeFormat("en-US", {
    month: short ? "short" : "long",
    day: "numeric",
    ...(short ? {} : { year: "numeric" }),
  }).format(date);
}

function number(value: number, decimals = 0) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: decimals,
    minimumFractionDigits: decimals,
  }).format(value);
}

function Logo() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#123449]">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          className="h-6 w-6"
          aria-hidden="true"
        >
          <path
            d="M3 17.5 8.5 12l4 3.5L21 6"
            stroke="#72E1C0"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M15.5 6H21v5.5"
            stroke="#72E1C0"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
      <div>
        <p className="text-lg font-bold tracking-tight text-white">
          Demand<span className="text-[#72E1C0]">Lens</span>
        </p>
        <p className="text-[11px] tracking-wide text-slate-400">
          SALES INTELLIGENCE
        </p>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  symbol,
}: {
  label: string;
  value: string;
  detail: string;
  symbol: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-50 text-lg text-teal-700">
          {symbol}
        </span>
      </div>
      <p className="mt-4 text-2xl font-semibold tracking-tight text-slate-900">
        {value}
      </p>
      <p className="mt-1 text-xs text-slate-500">{detail}</p>
    </div>
  );
}

export default function Home() {
  const [store, setStore] = useState("1");
  const [family, setFamily] = useState("BEVERAGES");
  const [horizon, setHorizon] = useState<Horizon>(7);
  const [forecast, setForecast] = useState<ForecastDay[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const metrics = useMemo(() => {
    if (!forecast?.length) return null;

    const total = forecast.reduce(
      (sum, day) => sum + day.predicted_sales,
      0
    );
    const peak = forecast.reduce((best, day) =>
      day.predicted_sales > best.predicted_sales ? day : best
    );

    return {
      total,
      average: total / forecast.length,
      peak,
    };
  }, [forecast]);

  async function generateForecast() {
    setLoading(true);
    setError("");
    setForecast(null);

    try {
      const response = await fetch(`${API_URL}/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          store_nbr: Number(store),
          family,
          start_date: "2017-08-16",
          horizon,
          assume_zero_promotions: true,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        const detail = data.detail;
        const message = Array.isArray(detail)
          ? detail.map((item: { msg?: string }) => item.msg).join("; ")
          : typeof detail === "string"
            ? detail
            : "The forecasting service rejected this request.";

        throw new Error(message);
      }

      const result = data as ForecastResponse;

      if (
        !Array.isArray(result.forecast) ||
        result.forecast.length !== horizon
      ) {
        throw new Error("The API returned an unexpected forecast response.");
      }

      setForecast(result.forecast);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to reach the forecasting service. Check that FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#F5F7FA] text-slate-900">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-[250px] flex-col bg-[#102A3A] px-5 py-7 lg:flex">
        <Logo />

        <p className="mb-3 mt-12 px-3 text-[10px] font-semibold tracking-[0.18em] text-slate-500">
          WORKSPACE
        </p>

        <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/10 px-3 py-3 text-sm font-medium text-white">
          <span className="text-[#72E1C0]">▥</span>
          Sales forecasting
        </div>

        <div className="mt-3 flex items-center gap-3 rounded-xl px-3 py-3 text-sm text-slate-400">
          <span>◈</span>
          Scenario planning
          <span className="ml-auto rounded-md bg-white/10 px-1.5 py-0.5 text-[9px]">
            SOON
          </span>
        </div>

        <div className="mt-auto rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="flex items-center gap-2 text-xs font-medium text-[#72E1C0]">
            <span className="h-2 w-2 rounded-full bg-[#72E1C0]" />
            MODEL WORKSPACE
          </div>
          <p className="mt-3 text-sm font-medium text-white">
            LightGBM Forecast
          </p>
          <p className="mt-1 text-xs leading-5 text-slate-400">
            Cluster-based model
            <br />
            Daily sales predictions
          </p>
          <div className="mt-4 border-t border-white/10 pt-3 text-[11px] text-slate-500">
            Historical dataset · 2013–2017
          </div>
        </div>
      </aside>

      <div className="lg:pl-[250px]">
        <header className="flex h-[72px] items-center justify-between border-b border-slate-200 bg-white px-5 sm:px-8">
          <div>
            <p className="text-xs text-slate-500">Workspace / Forecasting</p>
            <p className="mt-1 text-sm font-semibold">Demand overview</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-500 sm:inline-flex">
              Demo dataset
            </span>
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#DDF4EC] text-xs font-bold text-[#17634F]">
              DL
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-[1440px] px-5 py-8 sm:px-8 sm:py-10">
          <div className="flex flex-col justify-between gap-5 md:flex-row md:items-end">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.17em] text-teal-700">
                Forecast workspace
              </p>
              <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
                See demand ahead.
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
                Generate daily sales forecasts by store and product family.
                Explore the expected demand profile before making decisions.
              </p>
            </div>
            <div className="flex items-center gap-2 self-start rounded-full border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 md:self-auto">
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              LightGBM model
            </div>
          </div>

          <section className="mt-8 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <div className="flex flex-col gap-2 border-b border-slate-100 pb-5 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="font-semibold text-slate-900">
                  Configure your forecast
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  Choose the business segment and forecast window.
                </p>
              </div>
              <span className="text-xs text-slate-400">
                All fields are required
              </span>
            </div>

            <div className="mt-5 grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
              <label className="block">
                <span className="mb-2 block text-xs font-semibold text-slate-600">
                  Store number
                </span>
                <select
                  value={store}
                  onChange={(event) => setStore(event.target.value)}
                  className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-teal-600 focus:ring-2 focus:ring-teal-600/10"
                >
                  {Array.from({ length: 54 }, (_, i) => i + 1).map((n) => (
                    <option key={n} value={n}>
                      Store {n}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="mb-2 block text-xs font-semibold text-slate-600">
                  Product family
                </span>
                <select
                  value={family}
                  onChange={(event) => setFamily(event.target.value)}
                  className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-teal-600 focus:ring-2 focus:ring-teal-600/10"
                >
                  {FAMILIES.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </label>

              <div>
                <span className="mb-2 block text-xs font-semibold text-slate-600">
                  Forecast horizon
                </span>
                <div className="grid h-11 grid-cols-3 gap-1 rounded-xl bg-slate-100 p-1">
                  {HORIZONS.map((days) => (
                    <button
                      key={days}
                      type="button"
                      onClick={() => setHorizon(days)}
                      aria-pressed={horizon === days}
                      className={`rounded-lg text-sm font-medium transition ${
                        horizon === days
                          ? "bg-[#123449] text-white shadow-sm"
                          : "text-slate-500 hover:text-slate-900"
                      }`}
                    >
                      {days} days
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <span className="mb-2 block text-xs font-semibold text-slate-600">
                  Promotion assumption
                </span>
                <div className="flex h-11 items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-teal-100 text-xs text-teal-800">
                    ✓
                  </span>
                  <span className="text-sm text-slate-700">
                    Zero promotions
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-5 flex flex-col gap-4 border-t border-slate-100 pt-5 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs leading-5 text-slate-500">
                Forecast start: <strong>August 16, 2017</strong>
                <br />
                This demo uses historical training data ending August 15, 2017.
              </p>
              <button
                type="button"
                onClick={generateForecast}
                disabled={loading}
                className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-[#123449] px-5 text-sm font-semibold text-white shadow-sm transition hover:bg-[#1A465D] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Generating forecast...
                  </>
                ) : (
                  <>
                    Generate forecast
                    <span aria-hidden="true">→</span>
                  </>
                )}
              </button>
            </div>
          </section>

          {error && (
            <div
              role="alert"
              className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800"
            >
              <p className="font-semibold">Forecast could not be generated</p>
              <p className="mt-1 break-words">{error}</p>
              <p className="mt-2 text-xs">
                Check that your FastAPI server is running on port 8000.
              </p>
            </div>
          )}

          <section className="mt-8">
            <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Forecast results
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  {forecast
                    ? `${family} · Store ${store} · ${horizon}-day forecast`
                    : "Your generated predictions will appear here."}
                </p>
              </div>
              {forecast && (
                <span className="text-xs text-slate-500">
                  {formatDate(forecast[0].date)} –{" "}
                  {formatDate(forecast[forecast.length - 1].date)}
                </span>
              )}
            </div>

            <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
              <MetricCard
                label="Forecast sales"
                value={metrics ? number(metrics.total) : "—"}
                detail="Total predicted units over the selected horizon"
                symbol="Σ"
              />
              <MetricCard
                label="Average per day"
                value={metrics ? number(metrics.average, 1) : "—"}
                detail="Mean predicted daily sales"
                symbol="↗"
              />
              <MetricCard
                label="Peak demand day"
                value={
                  metrics ? number(metrics.peak.predicted_sales) : "—"
                }
                detail={
                  metrics
                    ? formatDate(metrics.peak.date)
                    : "Highest predicted daily sales"
                }
                symbol="◆"
              />
            </div>

            <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                <div>
                  <h3 className="font-semibold text-slate-900">
                    Daily sales forecast
                  </h3>
                  <p className="mt-1 text-xs text-slate-500">
                    Predicted sales units by date
                  </p>
                </div>
                {forecast && (
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span className="h-2.5 w-2.5 rounded-sm bg-teal-600" />
                    Predicted sales
                  </div>
                )}
              </div>

              <div className="mt-6 h-[300px] w-full">
                {loading ? (
                  <div className="flex h-full items-center justify-center">
                    <div className="text-center">
                      <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-slate-200 border-t-teal-700" />
                      <p className="mt-3 text-sm text-slate-500">
                        Running the forecasting model...
                      </p>
                    </div>
                  </div>
                ) : forecast ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={forecast}
                      margin={{ top: 8, right: 8, left: -18, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient
                          id="forecastFill"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="0%"
                            stopColor="#159A80"
                            stopOpacity={0.2}
                          />
                          <stop
                            offset="100%"
                            stopColor="#159A80"
                            stopOpacity={0.01}
                          />
                        </linearGradient>
                      </defs>
                      <CartesianGrid
                        stroke="#E9EEF2"
                        strokeDasharray="4 4"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="date"
                        tickFormatter={(value: string) =>
                          formatDate(value, true)
                        }
                        tick={{ fontSize: 11, fill: "#64748B" }}
                        tickLine={false}
                        axisLine={false}
                        minTickGap={24}
                      />
                      <YAxis
                        tickFormatter={(value: number) =>
                          number(value)
                        }
                        tick={{ fontSize: 11, fill: "#64748B" }}
                        tickLine={false}
                        axisLine={false}
                        width={55}
                      />
                      <Tooltip
                        labelFormatter={(value) =>
                          formatDate(String(value))
                        }
                        formatter={(value) => [
                          number(Number(value), 1),
                          "Predicted sales",
                        ]}
                        contentStyle={{
                          borderRadius: 12,
                          border: "1px solid #E2E8F0",
                          boxShadow: "0 8px 24px rgba(15,23,42,0.08)",
                          fontSize: 12,
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="predicted_sales"
                        name="Predicted sales"
                        stroke="#159A80"
                        strokeWidth={2.5}
                        fill="url(#forecastFill)"
                        activeDot={{
                          r: 5,
                          fill: "#159A80",
                          stroke: "#FFFFFF",
                          strokeWidth: 2,
                        }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50/70 px-5 text-center">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-50 text-xl text-teal-700">
                      ↗
                    </div>
                    <p className="mt-4 text-sm font-semibold text-slate-800">
                      Your forecast starts here
                    </p>
                    <p className="mt-2 max-w-sm text-xs leading-5 text-slate-500">
                      Choose a store, product family, and horizon above.
                      Generate a forecast to explore the predicted demand
                      curve and summary metrics.
                    </p>
                  </div>
                )}
              </div>

              {forecast && (
                <div className="mt-4 border-t border-slate-100 pt-4 text-xs leading-5 text-slate-500">
                  These are model predictions, not actual sales or a guarantee
                  of future demand. Promotions are assumed to be zero for
                  every forecast day.
                </div>
              )}
            </div>

            {forecast && (
              <div className="mt-5 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4 sm:px-6">
                  <div>
                    <h3 className="font-semibold text-slate-900">
                      Daily breakdown
                    </h3>
                    <p className="mt-1 text-xs text-slate-500">
                      Individual predictions returned by the API
                    </p>
                  </div>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">
                    {forecast.length} days
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-slate-50 text-xs text-slate-500">
                      <tr>
                        <th className="px-5 py-3 font-medium sm:px-6">
                          Date
                        </th>
                        <th className="px-5 py-3 font-medium">
                          Day
                        </th>
                        <th className="px-5 py-3 text-right font-medium">
                          Promotions
                        </th>
                        <th className="px-5 py-3 text-right font-medium sm:px-6">
                          Predicted sales
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {forecast.map((day) => (
                        <tr
                          key={day.date}
                          className="transition hover:bg-slate-50"
                        >
                          <td className="whitespace-nowrap px-5 py-3 font-medium text-slate-800 sm:px-6">
                            {formatDate(day.date)}
                          </td>
                          <td className="px-5 py-3 text-slate-500">
                            {new Intl.DateTimeFormat("en-US", {
                              weekday: "long",
                              timeZone: "UTC",
                            }).format(
                              new Date(`${day.date}T12:00:00Z`)
                            )}
                          </td>
                          <td className="px-5 py-3 text-right text-slate-500">
                            {number(day.onpromotion)}
                          </td>
                          <td className="px-5 py-3 text-right font-semibold tabular-nums text-slate-900 sm:px-6">
                            {number(day.predicted_sales, 2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>

          <footer className="mt-10 flex flex-col gap-2 border-t border-slate-200 py-6 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between">
            <p>DemandLens · Sales forecasting workspace</p>
            <p>LightGBM · Historical Favorita dataset</p>
          </footer>
        </main>
      </div>
    </div>
  );
}
