<template>
  <v-row>
    <v-col cols="8">
      <div>Selected Image: {{ selectedImageName }} {{ savedFiles.length }} / {{ files.length }}</div>
      <div class="controls-wrapper">
        <div class="image-controls">
          <v-btn icon @click="prevImage"><v-icon>mdi-chevron-left</v-icon></v-btn>
          <v-btn icon @click="nextImage"><v-icon>mdi-chevron-right</v-icon></v-btn>
        </div>
      </div>
      <v-img class="image-container" :src="selectedImage">
        <div class="image-buttons">
          <v-btn class="ok-button" color="green" icon @click="saveFile(selectedImageName)"><v-icon>mdi-check</v-icon></v-btn>
          <v-btn class="delete-button" color="red" icon @click="deleteFile(selectedImageName)"><v-icon>mdi-delete</v-icon></v-btn>
        </div>
      </v-img>
    </v-col>
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
            <v-list-item-content>
              {{ file }}
              <v-icon v-if="isSaved(file)" color="green">mdi-check</v-icon>
            </v-list-item-content>
          </v-list-item>
        </v-list>
      </v-card>
      <v-textarea v-model="captions" label="Enter text here" rows="10"></v-textarea>
    </v-col>
  </v-row>

</template>

<style>
.controls-wrapper{
  position: relative;
  top: calc(100vh/2 - 50px);
  z-index: 9999;
}
.image-container {
  max-height: 80vh;
  position: relative;
}
.scrollable-list {
  max-height: 350px;
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
  flex-direction: column;
  align-items: flex-end;
  padding: 10px;
}
.ok-button {
  margin-bottom: 30px;
}
.selected-file {
  background-color: #ddd;
}
</style>

<script setup>
import axios from 'axios';
import { onMounted, ref, nextTick } from 'vue';

const captions = ref('')

const selectedImage = ref('')
const selectedImageIndex = ref(0)

const files= ref([])
const savedFiles = ref([])
const imageDir = ref('F:\\ImageSet\\dump\\mobcup_output')

const selectedImageName = ref('')
const selectedIndex = ref(1)

const getImage = (imagePath,selectedIndexValue) => {
  const formData = new FormData();
  selectedImageName.value = imagePath;
  selectedIndex.value = selectedIndexValue;
  selectedImageIndex.value = files.value.indexOf(imagePath)
  formData.append('file_name', imageDir.value + '\\' + imagePath);
  axios.post('http://127.0.0.1:5000/file', formData, { responseType: 'blob' })
    .then(response => {
      const reader = new FileReader();
      reader.readAsDataURL(response.data);
      reader.onload = () => {
        const imageData = reader.result;
        selectedImage.value = imageData;
      };
    })
    .catch(error => {
      console.log(error);
    });
  const formDataForText = new FormData();
  const textFileName = imagePath.replace(/\.[^/.]+$/, '.txt');
  formDataForText.append('file_name', imageDir.value + '\\' + textFileName);
  axios.post('http://127.0.0.1:5000/file', formDataForText)
    .then(response => {
      captions.value = response.data
    })
    .catch(error => {
      console.log(error);
    });
  // Auto-scroll to the selected file
  goTo(`#file-${selectedIndex.value}`)
}

const saveFile = (fileName) => {
  const formData = new FormData();
  formData.append('file_name', imageDir.value + '\\' + fileName.split('.')[0] + '.txt');
  formData.append('content', captions.value);
  axios.post('http://127.0.0.1:5000/save', formData)
    .then(response => {
      console.log(response.data);
      savedFiles.value = [...response.data.saved_files];
      nextImage();
    })
    .catch(error => {
      console.log(error);
    });
  // Auto-scroll to the selected file
  goTo(`#file-${selectedIndex.value}`)
}

const deleteFile = (fileName) => {
  const formData = new FormData();
  formData.append('file_name', imageDir.value + '\\' + fileName);
  axios.post('http://127.0.0.1:5000/delete', formData)
    .then(response => {
      console.log(response.data);
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
    })
    .catch(error => {
      console.log(error);
    });
  // Auto-scroll to the selected file
  goTo(`#file-${selectedIndex.value}`)
}

const goTo = (selector) => {
  const element = document.querySelector(selector);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

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

onMounted(listFiles)

// Add event listeners for keyboard controls
window.addEventListener('keydown', (event) => {
  const focusedElement = document.activeElement;
  if (focusedElement.tagName !== 'TEXTAREA') {
    switch (event.key) {
      case 'ArrowLeft': // Left arrow
        prevImage();
        break;
      case 'ArrowRight': // Right arrow
        nextImage();
        break;
      case ' ': // Space
        saveFile(selectedImageName.value);
        break;
      case 'Delete': // Delete
        deleteFile(selectedImageName.value);
        break;
    }
  }
});
</script>