import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import pyvista as pv
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from io import BytesIO
import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import base64
import tempfile
import laspy
from geopy.distance import geodesic
import math
import ctypes
import time
import pyproj
from pathlib import Path
import logging
import platform

# Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title="Analyse Bathymétrique GEBCO")

# Configuration du logging pour le débogage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Empêcher la mise en veille de l'ordinateur
if platform.system() == "Windows":
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception as e:
        logger.warning(f"Impossible d'empêcher la mise en veille : {e}")

# Initialisation des variables de session
def init_session_state():
    defaults = {
        "polylines": [],
        "map_state": {"zoom": 8, "center": None},
        "selected_polyline": None,
        "sample_interval_m": 5,
        "profile_data": None,
        "tiff_data": None,
        "vtk_data": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Fonction pour charger un fichier GeoTIFF
@st.cache_data(show_spinner=False)
def load_geotiff(file_obj):
    try:
        with rasterio.open(file_obj) as dataset:
            data = dataset.read(1)
            transform = dataset.transform
            bounds = dataset.bounds
            crs = dataset.crs
            if not all([data is not None, transform, bounds, crs]):
                raise ValueError("Données GeoTIFF invalides ou incomplètes.")
            if data.size == 0 or np.all(np.isnan(data)):
                raise ValueError("Données bathymétriques vides ou toutes NaN.")
            logger.info(f"GeoTIFF chargé: Taille={data.shape}, CRS={crs}, Bounds={bounds}")
            return data, transform, bounds, crs
    except Exception as e:
        st.error(f"Erreur lors du chargement du GeoTIFF : {str(e)}")
        logger.error(f"Erreur GeoTIFF : {str(e)}")
        return None, None, None, None

# Fonction pour charger et ajouter un fichier VTK
def load_and_add_vtk(vtk_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vtk") as tmp:
            tmp.write(vtk_file.read() if hasattr(vtk_file, "read") else vtk_file.getbuffer())
            tmp_path = tmp.name
        logger.info(f"VTK: Fichier temporaire créé à {tmp_path}")

        mesh = pv.read(tmp_path)
        points = mesh.points
        if len(points) < 2:
            raise ValueError("Le fichier VTK doit contenir au moins 2 points.")
        logger.info(f"VTK: {len(points)} points lus, Premier point={points[0]}")

        transformer = pyproj.Transformer.from_crs("epsg:32738", "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(points[:, 0], points[:, 1])
        z = points[:, 2]
        logger.info(f"VTK: Conversion réussie, premier point (lon, lat) = ({lon[0]:.4f}, {lat[0]:.4f})")

        coords = list(zip(lon, lat))
        polyline_name = f"Polyligne_VTK_{len(st.session_state.polylines) + 1}"
        st.session_state.polylines.append({"name": polyline_name, "coords": coords})
        logger.info(f"VTK: Polyligne '{polyline_name}' ajoutée avec {len(coords)} points.")
        return lon, lat, z
    except Exception as e:
        st.error(f"Erreur lors du chargement du VTK : {str(e)}")
        logger.error(f"Erreur VTK : {str(e)}")
        return None, None, None
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"VTK: Fichier temporaire {tmp_path} supprimé.")

# Calcul de l'orientation d'une polyligne
def calculate_bearing(start, end):
    try:
        lon1, lat1 = math.radians(start[0]), math.radians(start[1])
        lon2, lat2 = math.radians(end[0]), math.radians(end[1])
        delta_lon = lon2 - lon1
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = math.degrees(math.atan2(x, y))
        bearing = (bearing + 360) % 360
        bearing = (bearing + 180) % 360
        return bearing
    except Exception as e:
        st.error(f"Erreur dans le calcul du bearing : {str(e)}")
        logger.error(f"Erreur bearing : {str(e)}")
        return 0.0

# Conversion de l'orientation en direction cardinale
def bearing_to_cardinal(bearing):
    try:
        directions = [
            "NORD", "NNE", "NE", "ENE", "EST", "ESE", "SE", "SSE",
            "SUD", "SSO", "SO", "OSO", "OUEST", "ONO", "NO", "NNO"
        ]
        index = int((bearing + 11.25) / 22.5) % 16
        return directions[index]
    except Exception as e:
        st.error(f"Erreur dans la conversion en cardinale : {str(e)}")
        logger.error(f"Erreur cardinale : {str(e)}")
        return "Inconnu"

# Extraction du profil bathymétrique avec interpolation IDW
def extract_polyline_profile(data, transform, coords, sample_interval=0.1, sampling_callback=None, idw_callback=None):
    try:
        logger.info(f"Extraction profil: Coords={coords}, Interval={sample_interval}")
        if data is None or transform is None:
            raise ValueError("Données bathymétriques ou transformation manquantes.")
        if len(coords) < 2:
            raise ValueError(f"Polyligne doit contenir au moins 2 points : {coords}")

        profile = []
        distances = []
        sample_points = []
        total_distance = 0
        pixel_coords = []

        # Conversion des coordonnées géographiques en indices de raster
        for lon, lat in coords:
            try:
                row, col = rasterio.transform.rowcol(transform, lon, lat)
                if row < 0 or col < 0 or row >= data.shape[0] or col >= data.shape[1]:
                    st.warning(f"Coordonnées ({lon}, {lat}) hors limites. Ignorées.")
                    logger.info(f"Coordonnées hors limites: ({lon}, {lat})")
                    continue
                pixel_coords.append((row, col))
            except Exception as e:
                st.error(f"Erreur conversion coordonnées ({lon}, {lat}) : {str(e)}")
                logger.error(f"Erreur conversion coordonnées ({lon}, {lat}) : {str(e)}")
                return np.array([]), np.array([]), coords, [], "Inconnu", "Inconnu"

        if len(pixel_coords) < 2:
            st.error(f"Moins de 2 points valides : {pixel_coords}")
            logger.error(f"Moins de 2 points valides : {pixel_coords}")
            return np.array([]), np.array([]), coords, [], "Inconnu", "Inconnu"

        # Calcul du nombre total d'échantillons
        total_samples = sum(max(2, int(geodesic(coords[i], coords[i+1]).kilometers / sample_interval)) 
                           for i in range(len(coords)-1))
        current_sample = 0
        start_time = time.time()

        # Échantillonnage le long des segments de la polyligne
        for i in range(len(pixel_coords) - 1):
            start_lon, start_lat = coords[i]
            end_lon, end_lat = coords[i + 1]
            segment_distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers
            num_points = max(2, int(segment_distance / sample_interval))
            t_values = np.linspace(0, 1, num_points)

            start_row, start_col = pixel_coords[i]
            end_row, end_col = pixel_coords[i + 1]

            for t in t_values:
                lon = start_lon + t * (end_lon - start_lon)
                lat = start_lat + t * (end_lat - start_lat)
                row = start_row + t * (end_row - start_row)
                col = start_col + t * (end_col - start_col)

                if 0 <= int(row) < data.shape[0] and 0 <= int(col) < data.shape[1]:
                    value = data[int(row), int(col)]
                    if np.isnan(value):
                        logger.info(f"Valeur NaN détectée à ({lon}, {lat})")
                else:
                    value = np.nan
                    st.warning(f"Point interpolé ({lon}, {lat}) hors limites.")
                    logger.info(f"Point interpolé hors limites: ({lon}, {lat})")

                profile.append(value)
                distances.append(total_distance + t * segment_distance)

                if not sample_points or total_distance + t * segment_distance >= sample_points[-1]["distance"] + sample_interval:
                    sample_points.append({
                        "lon": lon,
                        "lat": lat,
                        "depth": value,
                        "depth_positive": abs(value) if not np.isnan(value) else np.nan,
                        "distance": total_distance + t * segment_distance
                    })

                current_sample += 1
                if sampling_callback is not None and (current_sample % 100 == 0 or current_sample == total_samples):
                    elapsed_time = time.time() - start_time
                    time_per_sample = elapsed_time / current_sample if current_sample > 0 else 0
                    remaining_samples = total_samples - current_sample
                    sampling_callback(current_sample, total_samples, remaining_samples * time_per_sample)

            total_distance += segment_distance

        # Validation des données pour l'interpolation
        valid_mask = ~np.isnan([p["depth"] for p in sample_points])
        if not np.any(valid_mask):
            st.error("Aucune donnée valide pour l'interpolation IDW.")
            logger.error("Aucune donnée valide pour l'interpolation IDW.")
            return np.array(distances), np.array(profile), coords, sample_points, "Inconnu", "Inconnu"

        interp_distances = np.array([p["distance"] for p in sample_points])[valid_mask]
        interp_depths = np.array([p["depth"] for p in sample_points])[valid_mask]

        # Interpolation IDW
        sample_interval_m = sample_interval * 1000
        p = 1 + (sample_interval_m - 1) / 19
        resolution_factor = 1 + (sample_interval_m - 1) / 19
        resolution = int(len(interp_distances) * resolution_factor)
        new_distances = np.linspace(interp_distances.min(), interp_distances.max(), resolution)
        interp_profile = np.zeros(resolution)

        start_time_idw = time.time()
        for i, xd in enumerate(new_distances):
            weights = 1 / (np.abs(interp_distances - xd) ** p + 1e-10)
            weights /= weights.sum()
            interp_profile[i] = np.sum(weights * interp_depths)

            if idw_callback is not None and (i + 1) % 100 == 0 or i + 1 == resolution:
                elapsed_time_idw = time.time() - start_time_idw
                time_per_point = elapsed_time_idw / (i + 1) if i + 1 > 0 else 0
                remaining_points = resolution - (i + 1)
                idw_callback(i + 1, resolution, remaining_points * time_per_point)

        interp_sample_points = []
        for i, dist in enumerate(new_distances):
            if not interp_sample_points or dist >= interp_sample_points[-1]["distance"] + sample_interval:
                idx = np.argmin(np.abs(interp_distances - dist))
                interp_sample_points.append({
                    "lon": sample_points[idx]["lon"],
                    "lat": sample_points[idx]["lat"],
                    "depth": interp_profile[i],
                    "depth_positive": abs(interp_profile[i]) if not np.isnan(interp_profile[i]) else np.nan,
                    "distance": dist
                })

        # Calcul des directions
        bearing = calculate_bearing(coords[0], coords[-1]) if len(coords) >= 2 else 0.0
        start_direction = bearing_to_cardinal(bearing)
        opposite_bearing = (bearing + 180) % 360
        end_direction = bearing_to_cardinal(opposite_bearing)

        logger.info(f"Extraction terminée: {len(interp_sample_points)} points extraits")
        return np.array(distances), np.array(profile), coords, interp_sample_points, start_direction, end_direction
    except Exception as e:
        st.error(f"Erreur extraction profil : {str(e)}")
        logger.error(f"Erreur extraction : {str(e)}")
        return np.array([]), np.array([]), coords, [], "Inconnu", "Inconnu"

# Création de l'image bathymétrique pour la carte
def create_image_overlay(data, bounds):
    try:
        norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        cmap = plt.cm.viridis
        img_data = cmap(norm(data))[:, :, :3]
        img_data = (img_data * 255).astype(np.uint8)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            img_path = tmp_file.name
            fig, ax = plt.subplots()
            ax.imshow(img_data, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
            ax.axis("off")
            plt.savefig(img_path, bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close(fig)

        with open(img_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()

        os.remove(img_path)
        logger.info("Image bathymétrique créée avec succès")
        return encoded
    except Exception as e:
        st.error(f"Erreur création image bathymétrique : {str(e)}")
        logger.error(f"Erreur image overlay : {str(e)}")
        return None

# Exportation des données
def export_to_csv(all_points, filename):
    try:
        df = pd.DataFrame([{
            "Latitude": p["lat"],  # Latitude en premier
            "Longitude": p["lon"],  # Longitude en second
            "Depth (m)": p["depth_positive"],
            "Distance (km)": p["distance"]
        } for p in all_points])
        df.to_csv(filename, index=False)
        logger.info(f"CSV exporté : {filename}")
        return filename
    except Exception as e:
        st.error(f"Erreur exportation CSV : {str(e)}")
        logger.error(f"Erreur CSV : {str(e)}")
        return None

def export_to_las(all_points, filename):
    try:
        if not all_points:
            st.error("Aucun point à exporter dans le fichier LAS.")
            return None
        points = np.array([[p["lat"], p["lon"], p["depth_positive"]] for p in all_points])  # Lat, Lon, Depth
        las = laspy.create(point_format=2, file_version="1.2")
        las.y = points[:, 0]  # Latitude (y)
        las.x = points[:, 1]  # Longitude (x)
        las.z = points[:, 2]  # Depth
        las.write(filename)
        logger.info(f"LAS exporté : {filename}")
        return filename
    except Exception as e:
        st.error(f"Erreur exportation LAS : {str(e)}")
        logger.error(f"Erreur LAS : {str(e)}")
        return None

def export_to_vtk(all_points, filename):
    try:
        if not all_points:
            st.error("Aucun point à exporter dans le fichier VTK.")
            return None
        points = np.array([[p["lat"], p["lon"], p["depth_positive"]] for p in all_points])  # Lat, Lon, Depth
        polyline = pv.PolyData(points)
        polyline.lines = np.array([len(all_points), *range(len(all_points))])
        polyline["Depth (m)"] = [p["depth_positive"] for p in all_points]
        polyline.save(filename, binary=True)
        logger.info(f"VTK exporté : {filename}")
        return filename
    except Exception as e:
        st.error(f"Erreur exportation VTK : {str(e)}")
        logger.error(f"Erreur VTK : {str(e)}")
        return None

def create_zip_file(selected_polyline, bathymetry_data, transform, sample_interval):
    zip_buffer = BytesIO()
    try:
        total_steps = 3  # CSV, LAS, VTK pour une seule polyligne
        current_step = 0
        progress_bar_export = st.progress(0)
        progress_text_export = st.empty()

        # Définir les callbacks pour la barre de progression
        def sampling_callback(current, total, time_remaining):
            progress_text_export.text(f"Échantillonnage: {current}/{total} (Temps restant: {time_remaining:.2f}s)")

        def idw_callback(current, total, time_remaining):
            progress_text_export.text(f"Interpolation IDW: {current}/{total} (Temps restant: {time_remaining:.2f}s)")

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Récupérer la polyligne sélectionnée
            poly = next((p for p in st.session_state.polylines if p["name"] == selected_polyline), None)
            if not poly:
                st.error(f"Polyligne '{selected_polyline}' non trouvée.")
                return None

            name = poly["name"]
            coords = poly["coords"]
            if len(coords) < 2:
                st.error(f"Polyligne '{name}' a moins de 2 points. Ignorée.")
                return None

            distances, profile, _, all_points, _, _ = extract_polyline_profile(
                bathymetry_data, transform, coords, sample_interval=sample_interval,
                sampling_callback=sampling_callback, idw_callback=idw_callback
            )
            if not all_points:
                st.error(f"Aucun point extrait pour {name}. Ignorée.")
                logger.error(f"Aucun point extrait pour {name}. Coords: {coords}")
                return None

            for ext, export_func in [(".csv", export_to_csv), (".las", export_to_las), (".vtk", export_to_vtk)]:
                filename = f"{name}{ext}"
                file_path = export_func(all_points, filename)
                if file_path:
                    zipf.write(file_path, f"{name}/{filename}")
                    os.remove(file_path)
                current_step += 1
                progress = min(current_step / total_steps, 1.0)
                progress_bar_export.progress(progress)
                progress_text_export.text(f"Exportation : {progress*100:.1f}% ({current_step}/{total_steps} fichiers)")

        progress_text_export.text("Exportation terminée : 100%")
        zip_buffer.seek(0)
        logger.info("Fichier ZIP créé avec succès")
        return zip_buffer
    except Exception as e:
        st.error(f"Erreur création ZIP : {str(e)}")
        logger.error(f"Erreur ZIP : {str(e)}")
        return None

# Interface principale
def main():
    init_session_state()
    st.title("Analyse Bathymétrique GEBCO")
    st.markdown("Chargez un fichier GeoTIFF GEBCO et un fichier VTK (optionnel) pour tracer des polylignes et extraire des profils bathymétriques.")

    # Chargement des fichiers
    uploaded_file = st.file_uploader("Charger un fichier GeoTIFF GEBCO", type=["tif", "tiff"])
    vtk_file = st.file_uploader("Charger un fichier VTK (optionnel)", type=["vtk"])

    bathymetry_data, transform, bounds, crs = None, None, None, None
    if uploaded_file is not None:
        if st.session_state.tiff_data != uploaded_file.getbuffer():
            st.session_state.tiff_data = uploaded_file.getbuffer()
            with st.spinner("Chargement du GeoTIFF..."):
                bathymetry_data, transform, bounds, crs = load_geotiff(BytesIO(st.session_state.tiff_data))
        else:
            bathymetry_data, transform, bounds, crs = load_geotiff(BytesIO(st.session_state.tiff_data))

        if bathymetry_data is not None and transform is not None:
            st.success("GeoTIFF chargé avec succès !")
            st.info(f"Taille: {bathymetry_data.shape}, CRS: {crs}, Limites: {bounds}")

            # Chargement VTK
            lon, lat, z = None, None, None
            if vtk_file is not None and st.session_state.vtk_data != vtk_file.getbuffer():
                st.session_state.vtk_data = vtk_file.getbuffer()
                with st.spinner("Chargement du fichier VTK..."):
                    lon, lat, z = load_and_add_vtk(BytesIO(st.session_state.vtk_data))
                if lon is not None:
                    st.success(f"VTK chargé: {len(lon)} points convertis.")

            # Carte bathymétrique
            st.header("Carte Bathymétrique")
            st.markdown("**Instructions**: Clic gauche pour ajouter des points à la polyligne, clic droit pour terminer.")

            center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
            if st.session_state.map_state["center"] is None:
                st.session_state.map_state["center"] = center

            m = folium.Map(
                location=st.session_state.map_state["center"],
                zoom_start=st.session_state.map_state["zoom"],
                tiles="cartodbpositron",
                max_bounds=True
            )

            # Ajout du fond bathymétrique
            encoded_image = create_image_overlay(bathymetry_data, bounds)
            if encoded_image:
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{encoded_image}",
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    opacity=0.7,
                ).add_to(m)
            else:
                st.error("Échec de l'ajout du fond bathymétrique.")

            # Affichage des polylignes
            for poly in st.session_state.polylines:
                folium.PolyLine(
                    locations=[(lat, lon) for lon, lat in poly["coords"]],
                    color="blue",
                    weight=2.5,
                    opacity=1,
                    popup=f"Polyligne: {poly['name']}"
                ).add_to(m)

            # Ajout des marqueurs VTK
            if lon is not None:
                for lo, la, depth in zip(lon, lat, z):
                    folium.CircleMarker(
                        location=[la, lo],
                        radius=3,
                        color='red' if depth < 0 else 'green',
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"Profondeur: {depth:.2f} m"
                    ).add_to(m)

            # Ajout du plugin Draw
            folium.plugins.Draw(
                export=False,
                draw_options={
                    "polyline": {
                        "allowIntersection": False,
                        "shapeOptions": {
                            "color": "blue",
                            "weight": 2.5,
                            "opacity": 1,
                            "clickable": True,
                            "pointStyle": {
                                "radius": 2,
                                "color": "red",
                                "fillColor": "red",
                                "fillOpacity": 0.8
                            }
                        }
                    },
                    "polygon": False,
                    "circle": False,
                    "marker": False,
                    "circlemarker": False,
                    "rectangle": False
                }
            ).add_to(m)

            # CSS personnalisé pour réduire la taille des marqueurs si pointStyle ne fonctionne pas
            css = """
            <style>
            .leaflet-draw-edit-edit .leaflet-marker-icon {
                width: 4px !important;
                height: 4px !important;
                margin-left: -2px !important;
                margin-top: -2px !important;
            }
            </style>
            """
            m.get_root().html.add_child(folium.Element(css))

            output = st_folium(
                m,
                height=600,
                key="folium_map",
                center=st.session_state.map_state["center"],
                zoom=st.session_state.map_state["zoom"]
            )

            # Gestion des nouvelles polylignes
            if output.get("last_active_drawing"):
                coords = output["last_active_drawing"]["geometry"]["coordinates"]
                if coords and coords not in [p["coords"] for p in st.session_state.polylines]:
                    polyline_name = st.text_input("Nom de la polyligne :", key=f"name_{len(st.session_state.polylines)}")
                    if polyline_name and st.button("Valider la polyligne"):
                        st.session_state.polylines.append({"name": polyline_name, "coords": coords})
                        st.session_state.map_state["center"] = [output["center"]["lat"], output["center"]["lng"]]
                        st.session_state.map_state["zoom"] = output["zoom"]
                        st.success(f"Polyligne '{polyline_name}' enregistrée.")
                        logger.info(f"Nouvelle polyligne: {coords}")

            # Sélection et extraction du profil
            if st.session_state.polylines:
                st.header("Extraction du Profil Bathymétrique")
                with st.form(key="extraction_form"):
                    selected_polyline = st.selectbox(
                        "Choisir une polyligne :",
                        [p["name"] for p in st.session_state.polylines],
                        index=[p["name"] for p in st.session_state.polylines].index(st.session_state.selected_polyline)
                        if st.session_state.selected_polyline in [p["name"] for p in st.session_state.polylines]
                        else 0
                    )
                    sample_interval_m = st.slider(
                        "Pas d'échantillonnage (mètres)",
                        min_value=1,
                        max_value=20,
                        value=st.session_state.sample_interval_m,
                        step=1
                    )
                    submit_button = st.form_submit_button("Extraire le profil")

                    if submit_button:
                        st.session_state.selected_polyline = selected_polyline
                        st.session_state.sample_interval_m = sample_interval_m
                        sample_interval_km = sample_interval_m / 1000.0

                        selected_coords = next(p["coords"] for p in st.session_state.polylines if p["name"] == selected_polyline)
                        logger.info(f"Coordonnées sélectionnées pour {selected_polyline}: {selected_coords}")

                        progress_bar_sampling = st.progress(0)
                        progress_text_sampling = st.empty()
                        progress_bar_idw = st.progress(0)
                        progress_text_idw = st.empty()

                        def sampling_callback(current, total, time_remaining):
                            progress_bar_sampling.progress(min(current / total, 1.0))
                            progress_text_sampling.text(f"Échantillonnage: {current}/{total} (Temps restant: {time_remaining:.2f}s)")

                        def idw_callback(current, total, time_remaining):
                            progress_bar_idw.progress(min(current / total, 1.0))
                            progress_text_idw.text(f"Interpolation IDW: {current}/{total} (Temps restant: {time_remaining:.2f}s)")

                        distances, profile, coords, all_points, start_direction, end_direction = extract_polyline_profile(
                            bathymetry_data, transform, selected_coords, sample_interval=sample_interval_km,
                            sampling_callback=sampling_callback, idw_callback=idw_callback
                        )

                        st.session_state.profile_data = {
                            "distances": distances,
                            "profile": profile,
                            "coords": coords,
                            "all_points": all_points,
                            "start_direction": start_direction,
                            "end_direction": end_direction,
                            "polyline_name": selected_polyline,
                            "sample_interval_m": sample_interval_m
                        }

                # Affichage du profil
                if st.session_state.profile_data:
                    df_profile = pd.DataFrame({
                        "Distance (km)": st.session_state.profile_data["distances"],
                        "Profondeur (m)": st.session_state.profile_data["profile"]
                    })
                    if not df_profile.empty:
                        fig_2d = go.Figure()
                        fig_2d.add_trace(go.Scatter(
                            x=df_profile["Distance (km)"],
                            y=df_profile["Profondeur (m)"],
                            mode="lines",
                            name="Profondeur",
                            line=dict(color="#87CEEB", width=3),
                            fill="tozeroy",
                            fillcolor="rgba(135, 206, 235, 0.3)",
                            text=[f"Profondeur: {y:.2f} m" for y in df_profile["Profondeur (m)"]],
                            hoverinfo="text"
                        ))
                        depth_range = df_profile["Profondeur (m)"].max() - df_profile["Profondeur (m)"].min()
                        offset = depth_range * 0.1 if depth_range != 0 else 100
                        fig_2d.update_layout(
                            title=f"Profil Bathymétrique - {st.session_state.profile_data['polyline_name']}",
                            xaxis_title="Distance (km)",
                            yaxis_title="Profondeur (m)",
                            hovermode="closest",
                            plot_bgcolor="rgba(240, 248, 255, 0.8)",
                            paper_bgcolor="rgba(240, 248, 255, 0.8)",
                            font=dict(color="#333333"),
                            height=600,
                            margin=dict(l=50, r=50, t=100, b=50),
                            annotations=[
                                dict(
                                    x=df_profile["Distance (km)"].iloc[0],
                                    y=df_profile["Profondeur (m)"].iloc[0] + offset,
                                    text=st.session_state.profile_data["start_direction"],
                                    showarrow=False,
                                    xanchor="center",
                                    yanchor="bottom",
                                    font=dict(size=12, color="#4682B4")
                                ),
                                dict(
                                    x=df_profile["Distance (km)"].iloc[-1],
                                    y=df_profile["Profondeur (m)"].iloc[-1] + offset,
                                    text=st.session_state.profile_data["end_direction"],
                                    showarrow=False,
                                    xanchor="center",
                                    yanchor="bottom",
                                    font=dict(size=12, color="#4682B4")
                                )
                            ]
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)

                        st.subheader("Points Échantillonnés")
                        st.dataframe(pd.DataFrame([{
                            "Longitude": p["lon"],
                            "Latitude": p["lat"],
                            "Profondeur (m)": p["depth"],
                            "Distance (km)": p["distance"]
                        } for p in st.session_state.profile_data["all_points"]]))

            # Exportation du profil sélectionné
            if st.session_state.polylines:
                if st.button("Exporter le profil sélectionné (ZIP)"):
                    if bathymetry_data is None or transform is None:
                        st.error("Aucun fichier GeoTIFF chargé. Veuillez charger un fichier GeoTIFF avant d'exporter.")
                    else:
                        with st.spinner("Création du fichier ZIP..."):
                            zip_buffer = create_zip_file(
                                st.session_state.selected_polyline,
                                bathymetry_data,
                                transform,
                                st.session_state.sample_interval_m / 1000.0
                            )
                            if zip_buffer:
                                st.download_button(
                                    label="Télécharger ZIP",
                                    data=zip_buffer,
                                    file_name=f"{st.session_state.selected_polyline}_profile.zip",
                                    mime="application/zip"
                                )

# Nettoyage final
if __name__ == "__main__":
    try:
        main()
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)