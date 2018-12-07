#!/bin/bash
base_dir=$(pwd)
output_dir=${base_dir}
class_mapping_file=pascal_label_map.pbtxt

if [ -d "$output_dir" ]; then
  echo Dir existed
  # rm -r $output_dir
fi

echo find ${base_dir} -maxdepth 1 -type d ! -path .
DIRS=$(find ${base_dir} -maxdepth 1 -type d ! -path ${base_dir})

echo Concat ${class_mapping_file} files
rm ${output_dir}/${class_mapping_file}
for i in ${DIRS}; do
  echo $i
  echo "--"

  TRAIN_FILE=$(find ${i} -regextype sed -regex ".*./[0-9]\+_\(train\).txt")
  VAL_FILE=$(find ${i} -regextype sed -regex ".*./[0-9]\+_\(val\).txt")
  TOTAL_TRAIN_FILE=$(expr $(wc -l < $TRAIN_FILE) / 7) 
  TOTAL_VAL_FILE=$(wc -l < $VAL_FILE)
  echo $TOTAL_TRAIN_FILE $TOTAL_VAL_FILE
  if [[ "$TOTAL_TRAIN_FILE" -gt  "$TOTAL_VAL_FILE" ]]; then

	head -${TOTAL_TRAIN_FILE} $TRAIN_FILE >> $VAL_FILE
	tail -n +${TOTAL_TRAIN_FILE} $TRAIN_FILE > tmp.txt
	mv tmp.txt $TRAIN_FILE
  fi

  # remove duplicate line

  filename=${i}/${class_mapping_file}
  if [ ! -f "${i}/${class_mapping_file}" ]; then
    echo file not existed "${i}/${class_mapping_file}"
    rm -rf "${i}"
  fi
  head -4 $filename >> ${output_dir}/${class_mapping_file}
  echo "" >> ${output_dir}/${class_mapping_file}
done


FILES=$(find ${base_dir} -regextype sed -regex ".*./[0-9]\+_\(train\|val\).txt")


for file in ${FILES}; do
  sort ${file}| uniq -u > tmp.txt
  mv tmp.txt ${file}
  sed -i '/-1/d' ${file}
  sed -i '/^$/d' ${file}
done







