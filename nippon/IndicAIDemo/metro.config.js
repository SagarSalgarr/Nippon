const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');
const path = require('path');

const indicaiRoot = path.resolve(__dirname, '../react-native-indicai');

const config = {
  watchFolders: [indicaiRoot],
  resolver: {
    nodeModulesPaths: [
      path.resolve(__dirname, 'node_modules'),
      path.resolve(indicaiRoot, 'node_modules'),
    ],
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
