#!/bin/sh

video_file="$1"  # Chemin de la vidéo d'entrée
output_dir="/tmp/$1_frame"  # Chemin du dossier de sortie pour les images PNG

# Vérifier si le dossier de sortie existe, sinon le créer
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/png/"
    mkdir -p "$output_dir/tga/"
fi
if [ ! -d "/tmp/video_frame_mask/" ]; then
    mkdir -p "/tmp/video_frame_mask/"
fi

# Utiliser ffmpeg pour extraire les images de la vidéo au format PNG
ffmpeg -i "$video_file" -vf "fps=25" "$output_dir/png/frame_%04d.png"

echo "Start Mofrigy"

mogrify -monitor -format tga $output_dir/png/*.png
mv $output_dir/png/*.tga $output_dir/tga/

echo "End Mofrigy"

for file in $output_dir/tga/[0-99].tga; do
    new_name="0$file"
    mv "$file" "$new_name"
done
echo "END"
