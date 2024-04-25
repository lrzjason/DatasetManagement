<template>
  <div> 
    <v-app-bar app color="blue" dark>
      <v-toolbar-title>Dataset Management</v-toolbar-title>
      <!-- header content -->
      <!-- <v-menu offset-y>
        <template v-slot:activator="{ on, attrs }">
          <v-btn icon v-bind="attrs" v-on="on">
            <v-icon>mdi-menu</v-icon>
          </v-btn>
        </template>
        <v-list>
          <v-list-item to="/home"> <v-list-item-title>Home</v-list-item-title> </v-list-item>
          <v-list-item to="/check"> <v-list-item-title>Check</v-list-item-title> </v-list-item> 
        </v-list> 
      </v-menu> -->
      <v-spacer></v-spacer>
      <v-btn v-for="btn in pages" :key="btn.label" :to="btn.url">
        {{ btn.label }}
      </v-btn>
      <!-- <v-btn to="/home">
        <v-icon>mdi-magnify</v-icon>
        Home
      </v-btn>
      <v-btn to="/check">
        <v-icon>mdi-magnify</v-icon>
        Check
      </v-btn> -->
      <!-- <v-menu dark>
        <template v-slot:activator="{ props }">
          <v-btn
            dark
            color="white"
            v-bind="props"
          >
            Activator slot
          </v-btn>
        </template>
        <v-list>
          <v-list-item
            v-for="(item, index) in items"
            :key="index"
            :value="index"
          >
            <v-list-item-title>{{ item.title }}</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-menu> -->
    </v-app-bar> 
    <v-main>
      <slot />
    </v-main> 
    <v-footer app> <!-- footer content --> </v-footer> </div>
</template>

<script setup>
import { ref } from 'vue'

const pages = ref([
  {
    label:'File',
    url:'/'
  },
  {
    label:'Pairs',
    url:'/pairs'
  },
  {
    label:'Prepare',
    url:'/prepare'
  },
  {
    label:'Generate',
    url:'/generateImage'
  }
])
</script>