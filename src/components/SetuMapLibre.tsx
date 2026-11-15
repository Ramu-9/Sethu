import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import {
  LocationConfirmationStatus,
  SYNTHETIC_LANDMARKS,
  SimulatedAssignment,
  SyntheticLandmark,
} from '../utils/setuEngine';

export interface SetuMapLibreProps {
  latitude: number;
  longitude: number;
  accuracyMeters: number | null;
  matchConfidence?: number | null;
  locationSource: 'browser_gps' | 'landmark' | 'unresolved';
  confirmationStatus: LocationConfirmationStatus;
  matchedLandmarkId: string | null;
  callerLandmarkText?: string | null;
  broadLocationText?: string | null;
  activeAssignments?: SimulatedAssignment[];
  onSelectLandmark?: (landmark: SyntheticLandmark) => void;
}

/**
 * Generates a 64-vertex GeoJSON Polygon approximating a geodesic circle of `radiusMeters`
 * around `[lon, lat]`. Used strictly for GPS accuracy circles.
 */
interface GeoJsonCirclePolygonFeatureCollection {
  type: 'FeatureCollection';
  features: Array<{
    type: 'Feature';
    properties: { radiusMeters: number };
    geometry: {
      type: 'Polygon';
      coordinates: [number, number][][];
    };
  }>;
}

function createGeoJsonCircle(
  lon: number,
  lat: number,
  radiusMeters: number,
  points = 64,
): GeoJsonCirclePolygonFeatureCollection {
  const safeRadiusKm = Math.max(0.015, radiusMeters / 1000);
  const distanceX = safeRadiusKm / (111.32 * Math.cos((lat * Math.PI) / 180));
  const distanceY = safeRadiusKm / 110.574;

  const ring: [number, number][] = [];
  for (let i = 0; i < points; i++) {
    const theta = (i / points) * (2 * Math.PI);
    const x = distanceX * Math.cos(theta);
    const y = distanceY * Math.sin(theta);
    ring.push([lon + x, lat + y]);
  }
  ring.push(ring[0]);

  return {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        properties: { radiusMeters },
        geometry: {
          type: 'Polygon',
          coordinates: [ring],
        },
      },
    ],
  };
}

const EMPTY_GEOJSON_COLLECTION: GeoJsonCirclePolygonFeatureCollection = {
  type: 'FeatureCollection',
  features: [],
};

export const SetuMapLibre: React.FC<SetuMapLibreProps> = ({
  latitude,
  longitude,
  accuracyMeters,
  matchConfidence = null,
  locationSource,
  confirmationStatus,
  matchedLandmarkId,
  callerLandmarkText = null,
  broadLocationText = null,
  activeAssignments = [],
  onSelectLandmark,
}) => {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<maplibregl.Map | null>(null);
  const activeAssignmentsRef = useRef<SimulatedAssignment[]>(activeAssignments);
  activeAssignmentsRef.current = activeAssignments;
  const [mapLoaded, setMapLoaded] = useState<boolean>(false);
  const [pinPixelPos, setPinPixelPos] = useState<{ x: number; y: number } | null>(null);
  const [vehicleMarkers, setVehicleMarkers] = useState<
    Array<{
      id: string;
      code: string;
      category: string;
      status: string;
      eta: number | null;
      x: number;
      y: number;
    }>
  >([]);

  const matchedLandmark =
    SYNTHETIC_LANDMARKS.find((lm) => lm.landmark_id === matchedLandmarkId) ??
    SYNTHETIC_LANDMARKS.find(
      (lm) =>
        Math.abs(lm.latitude - latitude) < 0.0015 &&
        Math.abs(lm.longitude - longitude) < 0.0015,
    );

  // Accuracy circle and radius are shown ONLY for browser_gps
  const showGpsAccuracyRadius =
    locationSource === 'browser_gps' && accuracyMeters !== null;

  // Initialize real MapLibre GL JS instance with legible road + label tiles
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) {
      return;
    }

    const styleSpec: maplibregl.StyleSpecification = {
      version: 8,
      sources: {
        'carto-voyager': {
          type: 'raster',
          tiles: [
            'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png',
            'https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png',
          ],
          tileSize: 256,
          attribution: '© OpenStreetMap contributors, © CARTO',
        },
      },
      layers: [
        {
          id: 'carto-voyager-layer',
          type: 'raster',
          source: 'carto-voyager',
          minzoom: 0,
          maxzoom: 20,
        },
      ],
    };

    try {
      const map = new maplibregl.Map({
        container: mapContainerRef.current,
        style: styleSpec,
        center: [longitude, latitude],
        zoom: 15,
        attributionControl: false,
      });

      map.addControl(
        new maplibregl.NavigationControl({ showCompass: false }),
        'bottom-right',
      );

      const projectDemoVehicles = (mapRef: maplibregl.Map) => {
        const nextMarkers: Array<{
          id: string;
          code: string;
          category: string;
          status: string;
          eta: number | null;
          x: number;
          y: number;
        }> = [];
        for (const asgn of activeAssignmentsRef.current) {
          if (
            asgn.status === 'simulated_assigned' &&
            asgn.resource_id &&
            typeof asgn.current_latitude === 'number' &&
            typeof asgn.current_longitude === 'number'
          ) {
            const vPt = mapRef.project([asgn.current_longitude, asgn.current_latitude]);
            nextMarkers.push({
              id: asgn.assignment_id,
              code: asgn.vehicle_code || asgn.resource_id,
              category: asgn.category,
              status: asgn.lifecycle_status || 'ASSIGNED',
              eta: asgn.remaining_eta_minutes ?? asgn.simulated_eta_minutes,
              x: vPt.x,
              y: vPt.y,
            });
          }
        }
        setVehicleMarkers(nextMarkers);
      };

      const syncOverlayPosition = () => {
        const pt = map.project([longitude, latitude]);
        setPinPixelPos({ x: pt.x, y: pt.y });
        projectDemoVehicles(map);
      };

      map.on('load', () => {
        setMapLoaded(true);
        const initialCircleData =
          locationSource === 'browser_gps' && accuracyMeters !== null
            ? createGeoJsonCircle(longitude, latitude, accuracyMeters)
            : EMPTY_GEOJSON_COLLECTION;

        if (!map.getSource('caller-accuracy-circle')) {
          map.addSource('caller-accuracy-circle', {
            type: 'geojson',
            data: initialCircleData,
          });

          map.addLayer({
            id: 'caller-accuracy-fill',
            type: 'fill',
            source: 'caller-accuracy-circle',
            paint: {
              'fill-color':
                confirmationStatus === 'confirmed' ? '#10b981' : '#0d9488',
              'fill-opacity': 0.16,
            },
          });

          map.addLayer({
            id: 'caller-accuracy-outline',
            type: 'line',
            source: 'caller-accuracy-circle',
            paint: {
              'line-color':
                confirmationStatus === 'confirmed' ? '#059669' : '#0f766e',
              'line-width': 2,
              'line-dasharray': [2, 2],
            },
          });
        }
        syncOverlayPosition();
      });

      map.on('move', syncOverlayPosition);
      map.on('zoom', syncOverlayPosition);
      map.on('resize', syncOverlayPosition);

      mapInstanceRef.current = map;
      syncOverlayPosition();
    } catch {
      setMapLoaded(false);
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update MapLibre center & GeoJSON accuracy circle whenever coordinates, source, or status change
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) {
      return;
    }

    try {
      map.easeTo({
        center: [longitude, latitude],
        duration: 450,
      });
      const pt = map.project([longitude, latitude]);
      setPinPixelPos({ x: pt.x, y: pt.y });
      const nextMarkers: Array<{
        id: string;
        code: string;
        category: string;
        status: string;
        eta: number | null;
        x: number;
        y: number;
      }> = [];
      for (const asgn of activeAssignments) {
        if (
          asgn.status === 'simulated_assigned' &&
          asgn.resource_id &&
          typeof asgn.current_latitude === 'number' &&
          typeof asgn.current_longitude === 'number'
        ) {
          const vPt = map.project([asgn.current_longitude, asgn.current_latitude]);
          nextMarkers.push({
            id: asgn.assignment_id,
            code: asgn.vehicle_code || asgn.resource_id,
            category: asgn.category,
            status: asgn.lifecycle_status || 'ASSIGNED',
            eta: asgn.remaining_eta_minutes ?? asgn.simulated_eta_minutes,
            x: vPt.x,
            y: vPt.y,
          });
        }
      }
      setVehicleMarkers(nextMarkers);

      if (mapLoaded) {
        const src = map.getSource('caller-accuracy-circle') as
          | maplibregl.GeoJSONSource
          | undefined;
        if (src) {
          src.setData(
            locationSource === 'browser_gps' && accuracyMeters !== null
              ? createGeoJsonCircle(longitude, latitude, accuracyMeters)
              : EMPTY_GEOJSON_COLLECTION,
          );
        }
        if (map.getLayer('caller-accuracy-fill')) {
          map.setPaintProperty(
            'caller-accuracy-fill',
            'fill-color',
            confirmationStatus === 'confirmed'
              ? '#10b981'
              : confirmationStatus === 'candidate_awaiting_confirmation'
                ? '#0d9488'
                : '#dc2626',
          );
        }
        if (map.getLayer('caller-accuracy-outline')) {
          map.setPaintProperty(
            'caller-accuracy-outline',
            'line-color',
            confirmationStatus === 'confirmed'
              ? '#059669'
              : confirmationStatus === 'candidate_awaiting_confirmation'
                ? '#0f766e'
                : '#b91c1c',
          );
        }
      }
    } catch {
      // Ignore projection update errors during unmount
    }
  }, [latitude, longitude, accuracyMeters, locationSource, confirmationStatus, mapLoaded, activeAssignments]);

  // Visual radius in pixels for the translucent accuracy circle & pulsing outer ring (GPS only)
  const visualRadiusPx = !showGpsAccuracyRadius
    ? 14
    : accuracyMeters <= 100
      ? 38
      : accuracyMeters <= 500
        ? 52
        : 66;

  const pinBgClass =
    confirmationStatus === 'confirmed'
      ? 'bg-emerald-600'
      : confirmationStatus === 'candidate_awaiting_confirmation'
        ? 'bg-teal-600'
        : 'bg-red-600';

  const circleFillClass =
    confirmationStatus === 'confirmed'
      ? 'bg-emerald-500/20 border-emerald-600/70'
      : confirmationStatus === 'candidate_awaiting_confirmation'
        ? 'bg-teal-500/20 border-teal-600/70'
        : 'bg-red-500/20 border-red-600/70';

  const pulseRingClass =
    confirmationStatus === 'confirmed'
      ? 'border-emerald-500/60'
      : confirmationStatus === 'candidate_awaiting_confirmation'
        ? 'border-teal-500/60'
        : 'border-red-500/60';

  const primaryLabel =
    locationSource === 'landmark' && matchedLandmark
      ? matchedLandmark.name_romanised
      : callerLandmarkText
        ? callerLandmarkText
        : broadLocationText
          ? broadLocationText
          : matchedLandmark
            ? matchedLandmark.name_romanised
            : 'Hitech City / Madhapur Corridor';

  const resolvedMatchConfidence =
    matchConfidence ??
    (matchedLandmark ? matchedLandmark.match_confidence : 0.92);

  // Tooltip metric: GPS shows accuracy radius (±Xm), landmark shows match confidence (%)
  const tooltipMetricLabel =
    locationSource === 'browser_gps' && accuracyMeters !== null
      ? `±${Math.round(accuracyMeters)} m`
      : locationSource === 'landmark'
        ? `${Math.round(resolvedMatchConfidence * 100)}% match`
        : 'Unresolved';

  return (
    <div
      data-testid="spatial-coordination-map"
      className="w-full rounded-xl overflow-hidden border border-slate-200 bg-[#eef2f3] flex flex-col"
    >
      <div className="relative w-full h-64 sm:h-72 overflow-hidden">
      {/* Legible Vector-Style Fallback Road Network & Labels (visible while tiles stream or offline) */}
      <svg
        aria-hidden="true"
        viewBox="0 0 600 320"
        preserveAspectRatio="xMidYMid slice"
        className="absolute inset-0 w-full h-full pointer-events-none select-none"
      >
        <rect width="600" height="320" fill="#eef2f3" />
        {/* Urban blocks */}
        <rect x="20" y="20" width="160" height="90" rx="6" fill="#e2e8ea" />
        <rect x="210" y="18" width="190" height="92" rx="6" fill="#e2e8ea" />
        <rect x="430" y="22" width="150" height="88" rx="6" fill="#dde5e7" />
        <rect x="24" y="140" width="155" height="150" rx="6" fill="#e2e8ea" />
        <rect x="210" y="142" width="190" height="148" rx="6" fill="#e5ecee" />
        <rect x="430" y="140" width="148" height="152" rx="6" fill="#e2e8ea" />
        {/* Secondary roads */}
        <line x1="0" y1="70" x2="600" y2="70" stroke="#ffffff" strokeWidth="7" />
        <line x1="0" y1="220" x2="600" y2="220" stroke="#ffffff" strokeWidth="7" />
        <line x1="110" y1="0" x2="110" y2="320" stroke="#ffffff" strokeWidth="7" />
        <line x1="490" y1="0" x2="490" y2="320" stroke="#ffffff" strokeWidth="7" />
        {/* Primary arterial roads */}
        <line x1="0" y1="125" x2="600" y2="125" stroke="#fde68a" strokeWidth="12" />
        <line x1="0" y1="125" x2="600" y2="125" stroke="#ffffff" strokeWidth="8" />
        <line x1="195" y1="0" x2="195" y2="320" stroke="#fde68a" strokeWidth="12" />
        <line x1="195" y1="0" x2="195" y2="320" stroke="#ffffff" strokeWidth="8" />
        <line x1="415" y1="0" x2="415" y2="320" stroke="#ffffff" strokeWidth="10" />
        {/* Legible high-contrast road labels */}
        <text
          x="32"
          y="121"
          fill="#334155"
          fontSize="12"
          fontWeight="600"
          fontFamily="Plus Jakarta Sans, sans-serif"
        >
          NH-65 · Madhapur Main Rd
        </text>
        <text
          x="426"
          y="121"
          fill="#334155"
          fontSize="12"
          fontWeight="600"
          fontFamily="Plus Jakarta Sans, sans-serif"
        >
          Secunderabad Station Corridor
        </text>
        <text
          x="204"
          y="38"
          fill="#475569"
          fontSize="12"
          fontWeight="600"
          fontFamily="Plus Jakarta Sans, sans-serif"
        >
          Hitech City Rd
        </text>
      </svg>

      {/* Real MapLibre GL JS Interactive Container */}
      <div
        ref={mapContainerRef}
        className="absolute inset-0 w-full h-full"
        aria-label="MapLibre Incident Location Map"
      />

      {/* Top-Left Spec Compliance Pill: Synthetic Landmark Data Label */}
      <div className="absolute top-3 left-3 z-10 flex flex-wrap items-center gap-2 pointer-events-none">
        <span className="px-2.5 py-1 rounded-md bg-slate-900/90 text-white text-xs font-semibold shadow-sm">
          Synthetic landmark data
        </span>
        <span className="px-2.5 py-1 rounded-md bg-white/95 text-slate-800 border border-slate-300 text-xs font-medium tabular-nums shadow-sm">
          {latitude.toFixed(4)}° N, {longitude.toFixed(4)}° E
        </span>
      </div>

      {/* Synchronized Pin (20px marker with white ring), GPS-only Accuracy Circle, Pulsing Outer Ring, and Tooltip strictly ABOVE */}
      <div
        style={{
          left: pinPixelPos ? `${pinPixelPos.x}px` : '50%',
          top: pinPixelPos ? `${pinPixelPos.y}px` : '56%',
        }}
        className="absolute -translate-x-1/2 -translate-y-1/2 z-10 pointer-events-none flex items-center justify-center"
      >
        {/* Translucent Accuracy Circle — shown ONLY for browser_gps */}
        {showGpsAccuracyRadius && (
          <div
            data-testid="gps-accuracy-circle"
            style={{
              width: `${visualRadiusPx * 2}px`,
              height: `${visualRadiusPx * 2}px`,
            }}
            className={`rounded-full border-2 ${circleFillClass} flex items-center justify-center relative`}
          >
            {/* Pulsing Outer Ring (respects prefers-reduced-motion in index.css) */}
            <div
              className={`absolute inset-0 rounded-full border-2 ${pulseRingClass} setu-pulse-ring`}
            />
          </div>
        )}

        {/* Tooltip placed strictly ABOVE the pin/circle (or flipped below if pin is near top edge) */}
        <div
          style={
            pinPixelPos && pinPixelPos.y < 95
              ? { top: `${visualRadiusPx + 14}px` }
              : { bottom: `${visualRadiusPx + 14}px` }
          }
          className="absolute left-1/2 -translate-x-1/2 whitespace-nowrap bg-slate-900 text-white text-xs font-medium px-3 py-1.5 rounded-lg shadow-lg border border-slate-700 flex items-center gap-2"
        >
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${
              confirmationStatus === 'confirmed'
                ? 'bg-emerald-400'
                : confirmationStatus === 'candidate_awaiting_confirmation'
                  ? 'bg-amber-400'
                  : 'bg-red-400'
            }`}
          />
          <span className="font-semibold max-w-[195px] truncate" title={primaryLabel}>
            {primaryLabel}
          </span>
          <span
            data-testid="map-tooltip-metric"
            className="text-teal-200 font-semibold tabular-nums shrink-0"
          >
            {tooltipMetricLabel}
          </span>
          {/* Directional Arrow */}
          <span
            aria-hidden="true"
            className={
              pinPixelPos && pinPixelPos.y < 95
                ? 'absolute left-1/2 -top-1.5 -translate-x-1/2 w-3 h-3 bg-slate-900 border-l border-t border-slate-700 rotate-45'
                : 'absolute left-1/2 -bottom-1.5 -translate-x-1/2 w-3 h-3 bg-slate-900 border-r border-b border-slate-700 rotate-45'
            }
          />
        </div>

        {/* Clear 20px Marker with White Ring */}
        <div
          data-testid="map-caller-pin-20px"
          className={`absolute w-5 h-5 rounded-full ${pinBgClass} border-2 border-white shadow-md`}
        />
      </div>

      {/* Simulated Moving Demo Fleet Vehicle Pins */}
      {vehicleMarkers.map((vm, idx) => (
        <div
          key={vm.id}
          data-testid={`map-demo-vehicle-${vm.code}`}
          style={{
            left: `${vm.x + (idx % 3 - 1) * 14}px`,
            top: `${vm.y + Math.floor(idx / 3) * 14}px`,
          }}
          className="absolute -translate-x-1/2 -translate-y-1/2 z-10 pointer-events-none transition-all duration-700"
        >
          <div
            className={`px-2 py-0.5 rounded-md border shadow-md text-[10px] font-bold whitespace-nowrap flex items-center gap-1 ${
              vm.status === 'ARRIVED'
                ? 'bg-emerald-700 text-white border-emerald-200'
                : vm.status === 'EN_ROUTE'
                  ? 'bg-amber-600 text-white border-amber-200'
                  : 'bg-sky-700 text-white border-sky-200'
            }`}
          >
            <span>{vm.code}</span>
            <span className="opacity-90">· {vm.status}</span>
            {vm.eta !== null && <span className="opacity-90">({vm.eta}m)</span>}
          </div>
        </div>
      ))}
      </div>

      {/* Bottom Interactive Synthetic Landmark Quick-Selector Bar (Placed cleanly below map canvas) */}
      {onSelectLandmark && (
        <div className="flex items-center gap-1.5 overflow-x-auto py-1.5 px-2.5 bg-white border-t border-slate-200">
          <span className="text-xs font-bold text-slate-700 whitespace-nowrap pr-1">
            Landmarks:
          </span>
          {SYNTHETIC_LANDMARKS.map((lm) => {
            const isSelected =
              (matchedLandmarkId === lm.landmark_id ||
                matchedLandmark?.landmark_id === lm.landmark_id) &&
              locationSource === 'landmark';
            return (
              <button
                key={lm.landmark_id}
                type="button"
                onClick={() => onSelectLandmark(lm)}
                className={`min-h-[32px] px-2.5 py-1 rounded-md text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                  isSelected
                    ? 'bg-[#0B3B3C] text-white font-semibold shadow-2xs'
                    : 'bg-slate-100 text-slate-800 hover:bg-slate-200 border border-slate-200/80'
                }`}
              >
                {lm.name_romanised} ({Math.round(lm.match_confidence * 100)}%)
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};
