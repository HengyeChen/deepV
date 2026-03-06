#!/bin/bash
# DeepV pipeline parallel processing script
# Version: 1.0
# Usage: ./run_deepv_pipeline.sh -p <work_dir> -j <max_jobs> -b <bed_file>

# Function to display help information
show_help() {
    cat << EOF
DeepV Parallel Processing Pipeline Script

This script processes multiple chromosomes in parallel using DeepV pipeline.
It splits input BED file by chromosome, processes each chromosome independently,
and collects results in the final output directory.

Usage: $(basename "$0") -p <work_dir> -j <max_jobs> -b <bed_file>

Options:
  -p, --path      Working directory path (required)
  -j, --jobs      Maximum number of parallel jobs (default: 10)
  -b, --bed       Input BED file path (required)
  -h, --help      Display this help message

Input BED File Format:
  The input BED file should contain 4 columns:
    Column 1: Chromosome name (e.g., chrY)
    Column 2: Fragment midpoint coordinate
    Column 3: Fragment length
    Column 4: Fragment count number
  
  Example format:
    chrY    669491    94    1
    chrY    1169870   71    1
    chrY    1217535   61    1
    chrY    1411077   81    1
    chrY    2353754   67    1
    chrY    2781515   74    1

Docker Command Example:
  docker run -u \$(id -u):\$(id -g) \\
    -v /your_directory/result_dir:/data/final_results \\
    -v /your_directory/input_file_dir:/data/input_file \\      
    deepv:1.1 \\
    -p /data -j 10 -b /data/input_file/input.bed

EOF
    exit 0
}
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            WORK_PATH="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -b|--bed)
            BED_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$WORK_PATH" ]] || [[ -z "$BED_FILE" ]]; then
    echo "Error: Missing required parameters"
    echo "Use -h for help"
    exit 1
fi

# Set default values
: "${MAX_JOBS:=10}"

# Validate file existence
if [[ ! -f "$BED_FILE" ]]; then
    echo "Error: BED file not found: $BED_FILE"
    exit 1
fi

if [[ ! -f "hg38_chromsize.txt" ]]; then
    echo "Error: hg38_chromsize.txt not found in current directory"
    exit 1
fi

# Change to working directory
cd "$WORK_PATH" || {
    echo "Error: Cannot change to working directory: $WORK_PATH"
    exit 1
}

# Record start time
START_TIME=$(date +%s)
echo "================================================================================"
echo "DeepV Parallel Processing Pipeline"
echo "Start time: $(date)"
echo "================================================================================"
echo "Working directory: $WORK_PATH"
echo "Maximum parallel jobs: $MAX_JOBS"
echo "Input BED file: $BED_FILE"
echo "================================================================================"

# Activate Python environment
# if [[ -f "/mnt/disk2/2/zqf/private/DeepV/CNN/bin/activate" ]]; then
#     source "/mnt/disk2/2/zqf/private/DeepV/CNN/bin/activate"
#     echo "Python environment activated"
# else
#     echo "Warning: Python environment not found at specified path"
#     echo "Continuing with system Python environment"
# fi

# Create temporary directory for chromosome-specific BED files
TEMP_DIR="${WORK_PATH}/temp_bed_files"
mkdir -p "$TEMP_DIR"
echo "Created temporary directory: $TEMP_DIR"

# Extract chromosome information and convert to kb units
echo "Extracting chromosome information from hg38 reference..."
grep -E '^chr([1-9]|1[0-9]|2[0-2]|X|Y)[[:space:]]' hg38_chromsize.txt | \
awk 'BEGIN{OFS="\t"} {kb = int(($2+999999)/1000000)*1000; print $1, kb}' | \
sort -k1,1V > hg38_chromsize_kb.txt

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to process hg38_chromsize.txt"
    exit 1
fi

CHROM_COUNT=$(wc -l < hg38_chromsize_kb.txt)
echo "Found $CHROM_COUNT chromosomes to process"

# Split BED file by chromosome
echo "Splitting BED file by chromosome..."
grep -E '^chr([1-9]|1[0-9]|2[0-2]|X|Y)[[:space:]]' "${BED_FILE}" | \
awk -v outdir="$TEMP_DIR" '{print >> outdir"/"$1".bed"}'

# Count chromosomes that have data in BED file
BED_CHROM_COUNT=$(find "$TEMP_DIR" -name "*.bed" -type f | wc -l)
echo "Found $BED_CHROM_COUNT chromosomes with data in BED file"

# Process each chromosome in parallel
PROCESSED_COUNT=0
FAILED_CHROMS=()

echo "Starting parallel chromosome processing..."
echo "Maximum concurrent processes: $MAX_JOBS"
echo "================================================================================"

# Function to process a single chromosome
process_chromosome() {
    local chr="$1"
    local size="$2"
    local chrom_index="$3"
    
    echo "[$(date '+%H:%M:%S')] Starting chromosome $chr (${chrom_index}/$CHROM_COUNT)"
    
    # Create chromosome-specific log directory
    LOG_DIR="${WORK_PATH}/logs/${chr}"
    mkdir -p "$LOG_DIR"
    
    # Execute step1 processing
    cd "${WORK_PATH}/step1/" || return 1
    
    # Generate configuration
    python gen_config.py "${chr}" 0 "${size}" "${TEMP_DIR}/${chr}.bed" 2>&1 | tee -a "${LOG_DIR}/step1.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to generate config for $chr"
        return 1
    fi
    
    # Run conv1 processing
    python conv1.multithread.py "${chr}" 0 "${size}" 2>&1 | tee -a "${LOG_DIR}/step1.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: conv1 processing failed for $chr"
        return 1
    fi
    
    # Run conv2.V_inner processing
    python conv2.V_inner.multithread.py "${chr}" 0 "${size}" 2>&1 | tee -a "${LOG_DIR}/step1.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: conv2.V_inner processing failed for $chr"
        return 1
    fi
    
    # Get conv1 merge results
    python get_conv1merge.py "${chr}_0_${size}kb" 2>&1 | tee -a "${LOG_DIR}/step1.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to get conv1 merge for $chr"
        return 1
    fi
    
    # Run inference
    python infer_jointcost_focal_minfix_offset_chunked.py \
        --weights final_jointcost_weights.h5 \
        --input ${chr}_0_${size}kb/conv2/${chr}.conv1.*.merge.csv \
        --bed "${TEMP_DIR}/${chr}.bed" \
        --out "${chr}_0_${size}kb/model_infer/detected_points.tsv" \
        --thr 0.3 \
        --chunk_size 500 2>&1 | tee -a "${LOG_DIR}/step1.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: Inference failed for $chr"
        return 1
    fi
    
    # Move to step2 directory
    mv "${chr}_0_${size}kb" "${WORK_PATH}/step2/" 2>/dev/null
    
    # Execute step2 processing
    cd "${WORK_PATH}/step2/" || return 1
    
    # Filter detected points
    python filter_detected_points.py "${chr}_0_${size}kb/model_infer/detected_points.tsv" -o "${chr}_0_${size}kb" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to filter detected points for $chr"
        return 1
    fi
    
    # Count detected points
    local count
    count=$(wc -l ${chr}_0_${size}kb/conv2/${chr}.conv1.*.merge.csv 2>/dev/null | awk '{print $1-2}')
    
    if [[ -z "$count" ]] || [[ "$count" -le 0 ]]; then
        echo "Warning: No valid detected points for chromosome $chr, skipping step2"
        return 0
    fi
    
    # Generate step2 configuration
    python gen_config.py --result-dir "${chr}_0_${size}kb" \
        --config-dir "${chr}_0_${size}kb/config" \
        --detecte-point-path "${chr}_0_${size}kb/detected_points.filter.tsv" \
        --target-image-start 0 --target-image-end "${count}" \
        "${chr}" 0 "${size}" "${TEMP_DIR}/${chr}.bed" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to generate step2 config for $chr"
        return 1
    fi
    
    # Run step2 conv2.V_inner processing
    python conv2.V_inner.multithread.py "${chr}_0_${size}kb/config/conv2.0-${size}kb.config" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: step2 conv2.V_inner processing failed for $chr"
        return 1
    fi
    
    # Count detected points again
    count=$(wc -l ${chr}_0_${size}kb/conv2/detected_points.merge.csv 2>/dev/null | awk '{print $1-2}')
    
    if [[ -z "$count" ]] || [[ "$count" -le 0 ]]; then
        echo "Warning: No valid detected points after step2 for chromosome $chr, skipping further steps"
        return 0
    fi
    
    # Generate configuration for conv3
    python gen_config.py --result-dir "${chr}_0_${size}kb" \
        --config-dir "${chr}_0_${size}kb/config" \
        --detecte-point-path "${chr}_0_${size}kb/detected_points.filter.tsv" \
        --target-image-start 0 --target-image-end "${count}" \
        "${chr}" 0 "${size}" "${TEMP_DIR}/${chr}.bed" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to generate conv3 config for $chr"
        return 1
    fi
    
    # Run conv3 processing
    python conv3.new.multithread.py "${chr}_0_${size}kb/config/conv3.new.0-${size}kb.config" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: conv3 processing failed for $chr"
        return 1
    fi
    
    # Run conv2.V_channel processing
    python conv2.V_channel.multithread.py "${chr}_0_${size}kb/config/conv2.0-${size}kb.config" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: conv2.V_channel processing failed for $chr"
        return 1
    fi
    
    # Run post-conv2 max_min processing
    python post_conv2.max_min.oneconfig.multithread.batch.py "${chr}_0_${size}kb/config/post_conv2.0-${size}kb.config" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: post_conv2.max_min processing failed for $chr"
        return 1
    fi
    
    # Run post-conv2 pair processing
    python post_conv2.pair.multithread.batch.py "${chr}_0_${size}kb/config/post_conv2.0-${size}kb.config" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: post_conv2.pair processing failed for $chr"
        return 1
    fi
    
    # Run post-conv3 step1 processing
    python post_conv3.step1.multithread.py "${chr}_0_${size}kb/config/post_conv3.step1.0-${size}kb.config" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: post_conv3.step1 processing failed for $chr"
        return 1
    fi
    
    # Run post-conv3 step3 processing
    python post_conv3.step3.multithread.py "${chr}_0_${size}kb/config/post_conv3.step3.0-${size}kb.config" 2>&1 | tee -a "${LOG_DIR}/step2.log"
    if [[ $? -ne 0 ]]; then
        echo "Error: post_conv3.step3 processing failed for $chr"
        return 1
    fi
    
    echo "[$(date '+%H:%M:%S')] Completed chromosome $chr (${chrom_index}/$CHROM_COUNT)"
    return 0
}

# Export function for use in parallel processing
export -f process_chromosome
export WORK_PATH TEMP_DIR MAX_JOBS

# Read chromosome information and process in parallel
CHROM_INDEX=0
while read -r line; do
    arr=($line)
    chr="${arr[0]}"
    size="${arr[1]}"
    
    CHROM_INDEX=$((CHROM_INDEX + 1))
    
    # Check if BED file exists for this chromosome
    if [[ ! -f "${TEMP_DIR}/${chr}.bed" ]]; then
        echo "Skipping chromosome $chr: No data in BED file"
        continue
    fi
    
    # Process chromosome in background
    process_chromosome "$chr" "$size" "$CHROM_INDEX" &
    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    
    # Monitor and control parallel jobs
    while true; do
        # Count running Python processes
        RUNNING_JOBS=$(ps -ef | grep $(whoami)| grep -v grep | grep -c "python")
        
        if [[ $RUNNING_JOBS -lt $MAX_JOBS ]]; then
            break
        fi
        
        # Display status
        echo -ne "Active jobs: $RUNNING_JOBS/$MAX_JOBS | Processed: $PROCESSED_COUNT/$CHROM_COUNT\r"
        sleep 5
    done
    
done < hg38_chromsize_kb.txt

# Wait for all background processes to complete
echo "Waiting for all chromosome processing to complete..."
wait

echo "================================================================================"
echo "Chromosome processing completed"
echo "Processed: $PROCESSED_COUNT chromosomes"

# Create final results directory
FINAL_RESULTS_DIR="${WORK_PATH}/final_results"
mkdir -p "$FINAL_RESULTS_DIR"
mkdir -p ${FINAL_RESULTS_DIR}"/final_V_positions"
# Move chromosome results to final directory
if compgen -G "${WORK_PATH}/step2/chr*" > /dev/null; then
    mv "${WORK_PATH}/step2/chr"* "$FINAL_RESULTS_DIR/" 2>/dev/null
    echo "Results moved to: $FINAL_RESULTS_DIR"
	cp ${FINAL_RESULTS_DIR}/*/post_conv3/step3.filter/*topconv4.csv ${FINAL_RESULTS_DIR}/final_V_positions/
	echo "final results saved in final_V_positions/"
else
    echo "Warning: No chromosome results found in step2 directory"
fi

# Clean up temporary files
if [[ -d "$TEMP_DIR" ]]; then
    rm -rf "$TEMP_DIR"
    echo "Temporary files cleaned up"
fi

rm -f ${WORK_PATH}/step1/tmp* 2>/dev/null
# Calculate and display total runtime
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo "================================================================================"
echo "Pipeline completed successfully!"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "End time: $(date)"
echo "================================================================================"