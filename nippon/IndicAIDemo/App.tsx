import React from 'react';
import { StatusBar, SafeAreaView, StyleSheet } from 'react-native';
import IndicAIScreen from 'react-native-indicai/src/IndicAIScreen';

export default function App() {
  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="dark-content" backgroundColor="#F9FAFB" />
      <IndicAIScreen />
    </SafeAreaView>
  );
}
const styles = StyleSheet.create({ root: { flex: 1, backgroundColor: '#F9FAFB' } });
