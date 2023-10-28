<template>
  <v-row>
    
    <v-col cols="4">
      <h1>List of Files</h1>
      <v-card class="scrollable-list">
        <v-list>
          <v-list-item  
            :class="{ 'selected-file': selectedImageName === file }" 
            v-for="(file, index) in files" 
            :key="file" 
            @click="getImage(file, index)"
            :id="`file-${index}`"
          >
            <template v-slot:prepend>
              <v-icon v-if="isSaved(file)" color="green">mdi-check</v-icon>
            </template>

            <v-list-item-title>{{ file }}</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-card>
      <v-textarea v-model="captions" label="Enter text here" rows="10"></v-textarea>
    </v-col>
    <v-col cols="8">
      <div>Selected Image: {{ selectedImageName }} {{ savedFiles.length }} / {{ files.length }}</div>
      <div class="controls-wrapper">
        <div class="image-controls">
          <v-btn icon @click="prevImage"><v-icon>mdi-chevron-left</v-icon></v-btn>
          <v-btn icon @click="nextImage"><v-icon>mdi-chevron-right</v-icon></v-btn>
        </div>
      </div>
      <v-img ref="selectedImageRef" class="image-container" :src="imageSrc">
        <template v-slot:placeholder>
          <v-row
            class="fill-height ma-0"
            align="center"
            justify="center"
          >
            <v-progress-circular
              indeterminate
              color="grey-lighten-5"
            ></v-progress-circular>
          </v-row>
        </template>
        <div class="image-buttons">
          <v-btn class="ok-button" color="green" icon @click="saveFile(selectedImageName)"><v-icon>mdi-check</v-icon></v-btn>
          <v-btn class="delete-button" color="red" icon @click="deleteFile(selectedImageName)"><v-icon>mdi-delete</v-icon></v-btn>
        </div>
      </v-img>
    </v-col>
  </v-row>

</template>

<style>
.controls-wrapper{
  position: relative;
  top: calc(100vh/2 + 100px - 50px);
  z-index: 9999;
}
.image-container {
  max-height: 80vh;
  position: relative;
}
.scrollable-list {
  max-height: calc(100vh - 370px);
  overflow-y: auto;
}
.image-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.image-buttons {
  position: absolute;
  top: 0;
  right: 0;
  display: flex;
  flex-direction: row;
  align-items: flex-end;
  padding: 10px;
}
.ok-button {
  /* margin-bottom: 30px; */
  margin-right: 30px;
}
.selected-file {
  background-color: #ddd;
}
</style>

<script setup>
import axios from 'axios';
import { onMounted, ref, nextTick, onUnmounted } from 'vue';

const captions = ref('')

const selectedImage = ref('')
const selectedImageRef = ref(null)
const selectedImageIndex = ref(0)

const files= ref([])
const savedFiles = ref([])
const imageDir = ref('F:\\ImageSet\\dump\\mobcup_output')

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
const imageCache = [];


const getImage = (imagePath, selectedIndexValue) => {
  selectedImageName.value = imagePath;
  selectedIndex.value = selectedIndexValue;
  selectedImageIndex.value = files.value.indexOf(imagePath);
  const fileName = encodeURIComponent(imageDir.value + '/' + imagePath);
  console.log('get image', fileName);
  const cachedImageData = getCachedImageData(imagePath);
  if (cachedImageData) {
    imageSrc.value = cachedImageData;
  } else {
    const prevFiles = files.value.slice(Math.max(0, selectedImageIndex.value - 10), selectedImageIndex.value);
    const nextFiles = files.value.slice(selectedImageIndex.value + 1, selectedImageIndex.value + 11);
    const uncachedFiles = prevFiles.concat(nextFiles).filter(file => !imageCache.includes(file));
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
    axios.all(uncachedFiles.map(file => axios.get(`http://127.0.0.1:5000/file/${encodeURIComponent(imageDir.value + '/' + file)}`, { responseType: 'blob' })))
      .then(axios.spread((...responses) => {
        responses.forEach((response, index) => {
          const reader = new FileReader();
          reader.readAsDataURL(response.data);
          reader.onload = () => {
            const imageData = reader.result;
            imageCache.push(uncachedFiles[index]);
            if (imageCache.length > CACHE_SIZE) {
              imageCache.shift();
            }
            cacheImageData(uncachedFiles[index], imageData);
          };
        });
      }))
      .catch(error => {
        console.log(error);
      });
  }
  const textFileName = imagePath.replace(/\.[^/.]+$/, '.txt');
  const textFile = encodeURIComponent(imageDir.value + '/' + textFileName);
  axios.get(`http://127.0.0.1:5000/file/${textFile}`)
    .then(response => {
      captions.value = response.data
    })
    .catch(error => {
      console.log(error);
    })
  // Auto-scroll to the selected file
  goTo(`#file-${selectedIndex.value}`)
}


const getCachedImageData = (imagePath) => {
  const index = imageCache.findIndex((item) => item.path === imagePath);
  if (index !== -1) {
    const imageData = imageCache[index].data;
    // Move the cached image to the end of the array
    imageCache.splice(index, 1);
    imageCache.push({ path: imagePath, data: imageData });
    return imageData;
  }
  return null;
};

const cacheImageData = (imagePath, imageData) => {
  if (imageCache.length >= cacheSize) {
    // Remove the oldest cached image
    imageCache.shift();
  }
  imageCache.push({ path: imagePath, data: imageData });
};

// const getImage = (imagePath,selectedIndexValue) => {
//   const formData = new FormData();
//   selectedImageName.value = imagePath;
//   selectedIndex.value = selectedIndexValue;
//   selectedImageIndex.value = files.value.indexOf(imagePath)
//   const fileName = imageDir.value + '\\' + imagePath
//   console.log('get image', fileName);
//   selectedImageRef.value.loading = true;
//   formData.append('file_name', fileName);
//   axios.post('http://127.0.0.1:5000/file', formData, { responseType: 'blob' })
//     .then(response => {
//       const reader = new FileReader();
//       reader.readAsDataURL(response.data);
//       reader.onload = () => {
//         const imageData = reader.result;
//         selectedImage.value = imageData;
//       };
//     })
//     .catch(error => {
//       console.log(error);
//     }).finally(() => {
//       selectedImageRef.value.loading = false;
//     })
//   const formDataForText = new FormData();
//   const textFileName = imagePath.replace(/\.[^/.]+$/, '.txt');
//   formDataForText.append('file_name', imageDir.value + '\\' + textFileName);
//   axios.post('http://127.0.0.1:5000/file', formDataForText)
//     .then(response => {
//       captions.value = response.data
//     })
//     .catch(error => {
//       console.log(error);
//     })
//   // Auto-scroll to the selected file
//   goTo(`#file-${selectedIndex.value}`)
// }

const saveFile = debounce((fileName) => {
  console.log('save file', fileName)
  const formData = new FormData();
  formData.append('file_name', imageDir.value + '\\' + fileName.split('.')[0] + '.txt');
  formData.append('content', captions.value);
  // async save file
  nextImage();
  axios.post('http://127.0.0.1:5000/save', formData)
    .then(response => {
      console.log(response.data);
      savedFiles.value = [...response.data.saved_files];
    })
    .catch(error => {
      console.log(error);
    });
  // Auto-scroll to the selected file
  goTo(`#file-${selectedIndex.value}`)
},1000)

const deleteFile = debounce((fileName) => {
  const formData = new FormData();
  const sysFileName = imageDir.value+"\\"+fileName;
  console.log('delete file', sysFileName);
  formData.append('file_name', sysFileName);
  // async delete file
  const index = files.value.indexOf(fileName);
  if (index !== -1) {
    files.value.splice(index, 1);
    if (selectedImageName.value === fileName) {
      selectedImage.value = '';
      selectedImageName.value = '';
      captions.value = '';
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
  goTo(`#file-${selectedIndex.value}`)
},1000)

const goTo = (selector) => {
  const element = document.querySelector(selector);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

const prevImage = () => {
  // if (selectedImageIndex.value > 0) {
  //   getImage(files.value[selectedImageIndex.value - 1], selectedImageIndex.value - 1)
  // }
  const cachedImageData = getCachedImageData(files.value[selectedImageIndex.value - 1]);
  if (selectedImageIndex.value > 0 && cachedImageData) {
    selectedImageName.value = files.value[selectedImageIndex.value - 1];
    selectedIndex.value = selectedImageIndex.value - 1;
    selectedImageIndex.value--;
    imageSrc.value = cachedImageData;
    const textFileName = selectedImageName.value.replace(/\.[^/.]+$/, '.txt');
    const textFile = encodeURIComponent(imageDir.value + '/' + textFileName);
    axios
      .get(`http://127.0.0.1:5000/file/${textFile}`)
      .then((response) => {
        captions.value = response.data;
      })
      .catch((error) => {
        console.log(error);
      });
    // Auto-scroll to the selected file
    goTo(`#file-${selectedIndex.value}`);
  } else if (selectedImageIndex.value > 0) {
    getImage(files.value[selectedImageIndex.value - 1], selectedImageIndex.value - 1);
  }
}

const nextImage = () => {
  // if (selectedImageIndex.value < files.value.length - 1) {
  //   getImage(files.value[selectedImageIndex.value + 1], selectedImageIndex.value + 1)
  // }
  const cachedImageData = getCachedImageData(files.value[selectedImageIndex.value + 1]);
  if (selectedImageIndex.value < files.value.length - 1 && cachedImageData) {
    selectedImageName.value = files.value[selectedImageIndex.value + 1];
    selectedIndex.value = selectedImageIndex.value + 1;
    selectedImageIndex.value++;
    imageSrc.value = cachedImageData;
    const textFileName = selectedImageName.value.replace(/\.[^/.]+$/, '.txt');
    const textFile = encodeURIComponent(imageDir.value + '/' + textFileName);
    axios
      .get(`http://127.0.0.1:5000/file/${textFile}`)
      .then((response) => {
        captions.value = response.data;
      })
      .catch((error) => {
        console.log(error);
      });
    // Auto-scroll to the selected file
    goTo(`#file-${selectedIndex.value}`);
  } else if (selectedImageIndex.value < files.value.length - 1) {
    getImage(files.value[selectedImageIndex.value + 1], selectedImageIndex.value + 1);
  }
}

const isSaved = (fileName) => {
  fileName = imageDir.value+"\\"+fileName.split('.')[0] + '.txt';
  return savedFiles.value.includes(fileName);
}


const listFiles = () => {
  const formData = new FormData();
  formData.append('path', imageDir.value);
  axios.post('http://127.0.0.1:5000/list',formData)
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
  console.log(event.key);
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
  }
};

onMounted(()=>{
  listFiles();
  window.addEventListener('keydown', handleKeyboardControls);
})

onUnmounted(() => {
  unbindKeyboardControls();
});

</script>