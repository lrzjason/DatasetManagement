<template>
  <!-- <v-row>
    <v-col>
      <v-text-field v-model="imageDir" label="Target Dir">
      </v-text-field>
    </v-col>
  </v-row> -->
  <v-row>
    <v-col cols="3" v-show="!hideList">
      <h1>List of Files</h1>
      <v-card class="scrollable-list">
        <v-list>
          <v-list-item :class="{ 'selected-file': selectedImageName === file }" v-for="(file, index) in files" :key="file"
            @click="getImage(file, index)" :id="`file-${index}`">
            <template v-slot:prepend>
              <v-icon v-if="isSaved(file)" color="green">mdi-check</v-icon>
            </template>

            <v-list-item-title>{{ file }}</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-card>
      <v-textarea v-model="caption" label="Enter text here" rows="10"></v-textarea>
    </v-col>
    <v-col :cols="hideList ? 12 : 9">
      <div>Selected Image: {{ selectedImageName }} {{ selectedIndex }} / {{ files.length }}</div>
      <!-- <div class="controls-wrapper">
        <div class="image-controls">
          <v-btn icon @click="prevImage"><v-icon>mdi-chevron-left</v-icon></v-btn>
          <v-btn icon @click="nextImage"><v-icon>mdi-chevron-right</v-icon></v-btn>
        </div>
      </div> -->
      <div class="image-container">
        <v-img ref="selectedImageRef" max-height="700px" :class="hideList ? 'left200' : ''" :src="imageSrc">
          <template v-slot:placeholder>
            <v-row class="fill-height ma-0" align="center" justify="center">
              <v-progress-circular indeterminate color="grey-lighten-5"></v-progress-circular>
            </v-row>
          </template>
        </v-img>
        <div class="image-buttons">
          <v-btn 
            v-for="config in datasetConfig" 
            :key="config.label" 
            class="floating-button" 
            color="green"
            @click="saveToTarget(selectedImageName, config.label)">
            {{ config.label }}
          </v-btn>

          <v-btn class="floating-button" color="red" icon
            @click="deleteFile(selectedImageName)"><v-icon>mdi-delete</v-icon></v-btn>
          <v-btn class="floating-button" icon @click="prevImage"><v-icon>mdi-chevron-left</v-icon></v-btn>
          <v-btn class="end-button" icon @click="nextImage"><v-icon>mdi-chevron-right</v-icon></v-btn>
        </div>
      </div>
      <!-- <div class="caption-control">
        <v-textarea v-model="caption" hide-details variant="outlined" class="caption" label="caption" width="400px"
          auto-grow rows="1"></v-textarea>
      </div> -->
      <!-- <div class="caption-control-useful-words">
        <v-textarea v-model="captionUseful" hide-details variant="outlined" class="caption" label="caption" width="400px"
          auto-grow rows="1"></v-textarea>
      </div> -->

    </v-col>
  </v-row>
</template>

<style scoped>
.left200 {
  top:100px;
  left: -200px;
}

.image-el {
  max-height: calc(100vh - 110px);
  margin-top: 75px;
}

.caption textarea {
  text-align: center;
  font-size: 1.2em;
  color: white;
}

.image-classifier {
  color: white;
  position: absolute;
  z-index: 999;
}

.controls-wrapper {
  position: relative;
  top: calc(100vh/2 + 100px - 50px);
  z-index: 9999;
}

.image-container {
  display: flex;
  align-items: center;
  /* width: 512px; */
  max-height: 80vh;
}

.scrollable-list {
  max-height: calc(100vh - 450px);
  overflow-y: auto;
}

.image-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.caption-control-useful-words {
  background-color: rgba(0, 0, 0, 0.5);
  color: white;
  width: 600px;
  position: absolute;
  top: 100px;
  right: 100px;
  padding: 10px;
}

.caption-control {
  background-color: rgba(0, 0, 0, 0.5);
  color: white;
  width: 600px;
  position: absolute;
  bottom: 50px;
  right: 100px;
  padding: 10px;
}

.key-control {
  position: absolute;
  top: 70px;
  right: 50px;
  display: flex;
  flex-direction: row;
  align-items: flex-end;
  padding: 10px;
}

.image-buttons {
  position: absolute;
  top: 100px;
  right: 0;
  display: flex;
  flex-direction: row;
  align-items: flex-end;
  padding: 10px;
}

.floating-button {
  /* margin-bottom: 30px; */
  margin-right: 30px;
}

.end-button {
  margin-right: 50px;
}

.selected-file {
  background-color: #ddd;
}
</style>


<script setup>
import axios from 'axios';
import { onMounted, ref, nextTick, onUnmounted, watch } from 'vue';


const datasetConfig = ref([
  {
    label: 'worst',
    bindKey: 'h'
  },
  {
    label: 'bad',
    bindKey: 'j'
  },
  {
    label: 'good',
    bindKey: 'k'
  },
  {
    label: 'amazing',
    bindKey: 'l'
  },
  
])

const hideList = ref(true)
const caption = ref('')
// const captionUseful = ref('')

const selectedImage = ref('')
const selectedImageRef = ref(null)
const selectedImageIndex = ref(0)

const files = ref([])
const savedFiles = ref([])
const imageDir = ref('F:\\ImageSet\\pickscore_random_captions_pag_ays_parent\\pickscore_random_captions_pag_ays')
const saveToDir = ref('F:\\ImageSet\\pickscore_random_captions_train')
const selfRemove = ref(false)

// watch imageDir change, list files
watch(imageDir, (newValue, oldValue) => {
  listFiles();
})

const selectedImageName = ref('')
const selectedIndex = ref(1)

const debounce = (func, delay) => {
  let timeoutId;
  return (...args) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      func.apply(null, args);
    }, delay);
  };
};

const imageSrc = ref('')

const CACHE_SIZE = 20;
const imageCacheList = [];


const cacheImageData = (imagePath, imageData) => {
  if (imageCacheList.length >= CACHE_SIZE) {
    // Remove the oldest cached image
    imageCacheList.shift();
  }
  imageCacheList.push({ path: imagePath, data: imageData });
};

const cachingFiles = ref([])

const getImage = (imagePath, selectedIndexValue) => {
  const currentBatchCacheFiles = []

  selectedImageName.value = imagePath;
  selectedIndex.value = selectedIndexValue;
  selectedImageIndex.value = files.value.indexOf(imagePath);
  const fileName = encodeURIComponent(imageDir.value + '/' + imagePath);
  console.log('get image', fileName);
  const cachedImageData = getCachedImageData(imagePath);
  if (cachedImageData) {
    console.log('has cache', imagePath);
    imageSrc.value = cachedImageData;
  } else {
    axios.get(`http://127.0.0.1:5000/file/${fileName}`, { responseType: 'blob' })
      .then(response => {
        const reader = new FileReader();
        reader.readAsDataURL(response.data);
        reader.onload = () => {
          const imageData = reader.result;
          imageSrc.value = imageData;
          cacheImageData(imagePath, imageData);
        };
      })
      .catch(error => {
        console.log(error);
      });
  }
  // const textFileName = imagePath.replace(/\.[^/.]+$/, '.txt');
  // const textFile = encodeURIComponent(imageDir.value + '/' + textFileName);
  // axios.get(`http://127.0.0.1:5000/file/${textFile}`)
  //   .then(response => {
  //     caption.value = response.data
  //   })
  //   .catch(error => {
  //     console.log(error);
  //   })
  // Auto-scroll to the selected file
  // goTo(`#file-${selectedIndex.value}`)
  const prevFiles = files.value.slice(Math.max(0, selectedImageIndex.value - 10), selectedImageIndex.value);
  const nextFiles = files.value.slice(selectedImageIndex.value + 1, selectedImageIndex.value + 11);
  const uncachedFiles = prevFiles.concat(nextFiles).filter(file => imageCacheList.findIndex((item) => item.path === file) === -1);

  // push uncached files to cachingFiles if not already in cachingFiles
  uncachedFiles.forEach(file => {
    if (cachingFiles.value.findIndex(item => item === file) === -1) {
      currentBatchCacheFiles.push(file)
      cachingFiles.value.push(file)
    } else {
      console.log('already in cachingFiles', file)
    }
  })
  console.log('currentBatchCacheFiles', currentBatchCacheFiles)
  axios.all(currentBatchCacheFiles.map(file => axios.get(`http://127.0.0.1:5000/file/${encodeURIComponent(imageDir.value + '/' + file)}`, { responseType: 'blob' })))
    .then(axios.spread((...responses) => {
      responses.forEach((response, index) => {
        const reader = new FileReader();
        reader.readAsDataURL(response.data);
        reader.onload = () => {
          const imageData = reader.result;
          // imageCache.push(uncachedFiles[index]);
          if (imageCacheList.length > CACHE_SIZE) {
            imageCacheList.shift();
          }
          cacheImageData(uncachedFiles[index], imageData);
        };
        // remove file from cachingFiles
        const fileIndex = cachingFiles.value.findIndex(item => item === uncachedFiles[index])
        if (fileIndex !== -1) {
          console.log('remove from cachingFiles', uncachedFiles[index])
          cachingFiles.value.splice(fileIndex, 1)
        }
      });
    }))
    .catch(error => {
      console.log(error);
    });
}


const getCachedImageData = (imagePath) => {
  const index = imageCacheList.findIndex((item) => item.path === imagePath);
  if (index !== -1) {
    const imageData = imageCacheList[index].data;
    // Move the cached image to the end of the array
    // imageCache.splice(index, 1);
    // imageCache.push({ path: imagePath, data: imageData });
    return imageData;
  }
  return null;
};

const saveToTarget = debounce((fileName, label) => {
  // Auto-scroll to the selected file
  // goTo(`#file-${selectedIndex.value + 1}`)
  console.log('save to', fileName)
  const index = files.value.indexOf(fileName);
  const formData = new FormData();
  formData.append('label', label)
  formData.append('file_name', fileName)
  formData.append('save_from', imageDir.value)
  formData.append('save_to', saveToDir.value)
  formData.append('self_remove', selfRemove.value)
  axios.post('http://127.0.0.1:5000/save_to_target', formData)
    .then(response => {
      console.log(response.data);
      // savedFiles.value = [...response.data.saved_files];
      if (selfRemove.value){
        if (index !== -1) {
          files.value.splice(index, 1);
          if (selectedImageName.value === fileName) {
            selectedImage.value = '';
            selectedImageName.value = '';
            caption.value = '';
            if (index < files.value.length) {
              getImage(files.value[index], index);
            } else if (index > 0) {
              getImage(files.value[index - 1], index - 1);
            }
            selectedImageIndex.value--;
          } else if (index < selectedImageIndex.value) {
            selectedImageIndex.value--;
          }
        }
      }
    })
    .catch(error => {
      console.log(error);
    });
  // async save file
  nextImage();

}, 1000)

const saveFile = debounce((fileName) => {
  // Auto-scroll to the selected file
  // goTo(`#file-${selectedIndex.value + 1}`)
  // console.log('save file', fileName)
  // const formData = new FormData();
  // formData.append('file_name', imageDir.value + '\\' + fileName.split('.')[0] + '.txt');
  // formData.append('content', caption.value);
  // axios.post('http://127.0.0.1:5000/save', formData)
  //   .then(response => {
  //     console.log(response.data);
  //     savedFiles.value = [...response.data.saved_files];
  //   })
  //   .catch(error => {
  //     console.log(error);
  //   });
  // // async save file
  nextImage();

}, 1000)

const deleteFile = debounce((fileName) => {
  const formData = new FormData();
  const sysFileName = imageDir.value + "\\" + fileName;
  console.log('delete file', sysFileName);
  formData.append('file_name', sysFileName);
  // async delete file
  const index = files.value.indexOf(fileName);
  if (index !== -1) {
    files.value.splice(index, 1);
    if (selectedImageName.value === fileName) {
      selectedImage.value = '';
      selectedImageName.value = '';
      caption.value = '';
      if (index < files.value.length) {
        getImage(files.value[index], index);
      } else if (index > 0) {
        getImage(files.value[index - 1], index - 1);
      }
    } else if (index < selectedImageIndex.value) {
      selectedImageIndex.value--;
    }
  }
  axios.post('http://127.0.0.1:5000/delete', formData)
    .then(response => {
      console.log(response.data);

    })
    .catch(error => {
      console.log(error);
    })
  // Auto-scroll to the selected file
  // goTo(`#file-${selectedIndex.value}`)
}, 1000)

// const goTo = (selector) => {
//   const element = document.querySelector(selector);
//   if (element) {
//     element.scrollIntoView({ behavior: 'smooth', block: 'center' });
//   }
// }

const prevImage = () => {
  if (selectedImageIndex.value > 0) {
    getImage(files.value[selectedImageIndex.value - 1], selectedImageIndex.value - 1)
  }
}

const nextImage = () => {
  if (selectedImageIndex.value < files.value.length - 1) {
    getImage(files.value[selectedImageIndex.value + 1], selectedImageIndex.value + 1)
  }
}

const isSaved = (fileName) => {
  return savedFiles.value.includes(fileName);
}


const listFiles = () => {
  const formData = new FormData();
  formData.append('path', imageDir.value);
  axios.post('http://127.0.0.1:5000/list_dataset', formData)
    .then(response => {
      files.value = response.data.files;
      nextTick(() => {
        savedFiles.value = [...response.data.saved_files];
        if (files.value.length > 0) {
          const unsavedIndex = files.value.findIndex(file => !isSaved(file));
          if (unsavedIndex !== -1) {
            getImage(files.value[unsavedIndex], unsavedIndex);
          } else {
            getImage(files.value[0], 0);
          }
        }
      })
    })
    .catch(error => {
      console.log(error);
    });
}


const unbindKeyboardControls = () => {
  window.removeEventListener('keydown', handleKeyboardControls);
};

const handleKeyboardControls = (event) => {
  // console.log('trigger key event',event.key);
  const focusedElement = document.activeElement;
  if (focusedElement.tagName !== 'TEXTAREA') {
    switch (event.key) {
      case 'ArrowLeft': // Left arrow
        prevImage();
        break;
      case 'ArrowRight': // Right arrow
        nextImage();
        break;
      case 'End': // F1
        saveFile(selectedImageName.value);
        event.preventDefault();
        break;
      case 'Delete': // F2
        deleteFile(selectedImageName.value);
        event.preventDefault();
        break;
    }
    for (let index = 0; index < datasetConfig.value.length; index+=1) {
      const config = datasetConfig.value[index];
      if(event.key === config.bindKey){
        // console.log('trigger keyboard event: ',config.bindKey)
        saveToTarget(selectedImageName.value,config.label)
        break
      }
    }
  }
};

onMounted(() => {
  listFiles();
  window.addEventListener('keydown', handleKeyboardControls);
})

onUnmounted(() => {
  unbindKeyboardControls();
});

</script>